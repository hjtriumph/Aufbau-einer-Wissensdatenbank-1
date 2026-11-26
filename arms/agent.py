from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.documents import Document

from core.citations import (
    detect_refusal,
    format_context,
    parse_citation_labels,
    refusal_text,
    resolve_citations,
)
from core.llm import complete
from core.store import retrieve
from prompts.templates import REACT_FORCE_FINAL, REACT_STEP_USER, REACT_SYSTEM

from .pipeline import RetrievalBundle, add_usage, empty_usage
from .schema import ArmConfig

_ACTION_RE = re.compile(r"行动\s*[：:]\s*search\s*[（(]\s*(.+?)\s*[)）]", re.IGNORECASE)
_FINAL_RE = re.compile(r"最终答案\s*[：:]\s*(.*)", re.DOTALL)


def _doc_id(doc: Document) -> str:
    return str((doc.metadata or {}).get("chunk_id") or "")


def _format_observations(docs: list[Document], start: int) -> str:
    if not docs:
        return "（未检索到相关块）"
    blocks: list[str] = []
    for offset, doc in enumerate(docs, start=start):
        meta = doc.metadata or {}
        page = meta.get("page_start", meta.get("page", "?"))
        end = meta.get("page_end", page)
        page_label = f"{page}" if end in (None, page) else f"{page}-{end}"
        title = meta.get("doc_title") or meta.get("doc") or "unknown"
        blocks.append(f"[C{offset + 1}] {title} | page={page_label}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _parse_step(text: str) -> tuple[str, str]:
    final = _FINAL_RE.search(text or "")
    action = _ACTION_RE.search(text or "")
    # An explicit final answer wins: models often echo the action format while
    # answering, and treating that as a search would waste a step.
    if final:
        return "final", final.group(1).strip()
    if action:
        return "search", action.group(1).strip().strip("\"'“”")
    if text.strip():
        return "final", text.strip()
    return "final", ""


def run_agent(
    arm: ArmConfig,
    question: str,
    language: str = "zh",
    cached: RetrievalBundle | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    usage = empty_usage()
    docs: list[Document] = []
    seen: set[str] = set()
    candidate_ids: list[str] = []
    steps: list[dict[str, Any]] = []
    queries: list[str] = []
    answer = ""
    gen_latency = 0.0

    max_steps = max(1, arm.agent.max_steps)
    budget = max(0, arm.agent.max_chunks)

    for step in range(max_steps):
        remaining = budget - len(docs)
        observations = _format_observations(docs, 0) if docs else "（尚未检索）"
        if remaining <= 0:
            # Budget exhausted: force an answer instead of letting the model
            # spend another turn on a search that cannot return anything.
            user = REACT_FORCE_FINAL.format(question=question, observations=observations)
            text, step_usage, step_latency = complete(
                [("system", REACT_SYSTEM), ("human", user)]
            )
            usage = add_usage(usage, step_usage)
            gen_latency += step_latency
            match = _FINAL_RE.search(text)
            answer = match.group(1).strip() if match else text.strip()
            steps.append({"step": step + 1, "type": "final_forced", "output": text[:500]})
            break

        user = REACT_STEP_USER.format(question=question, observations=observations)
        text, step_usage, step_latency = complete(
            [("system", REACT_SYSTEM), ("human", user)]
        )
        usage = add_usage(usage, step_usage)
        gen_latency += step_latency

        kind, payload = _parse_step(text)
        if kind == "final":
            answer = payload
            steps.append({"step": step + 1, "type": "final", "output": text[:500]})
            break

        query = payload or question
        queries.append(query)
        top_k = min(arm.agent.per_step_k, remaining)
        result = retrieve(
            strategy=arm.retrieval.strategy,
            query=query,
            mode="dense",
            k=top_k,
            fetch_k=max(arm.retrieval.fetch_k, top_k),
        )
        candidate_ids.extend(_doc_id(d) for d in result.candidates)
        new_docs = []
        for doc in result.docs:
            doc_id = _doc_id(doc)
            # Budget counts blocks, not attempts: a repeated chunk must not
            # consume a second slot.
            if doc_id in seen:
                continue
            seen.add(doc_id)
            new_docs.append(doc)
            if len(docs) + len(new_docs) >= budget:
                break
        docs.extend(new_docs)
        steps.append(
            {
                "step": step + 1,
                "type": "search",
                "query": query,
                "new_chunks": len(new_docs),
                "total_chunks": len(docs),
            }
        )

    if not answer:
        observations = _format_observations(docs, 0) if docs else "（尚未检索）"
        text, final_usage, final_latency = complete(
            [
                ("system", REACT_SYSTEM),
                ("human", REACT_FORCE_FINAL.format(question=question, observations=observations)),
            ]
        )
        usage = add_usage(usage, final_usage)
        gen_latency += final_latency
        answer = _FINAL_RE.search(text).group(1).strip() if _FINAL_RE.search(text) else text.strip()

    if not docs and arm.output.refusal:
        answer = refusal_text(language)

    context, citation_map = format_context(docs)
    labels = parse_citation_labels(answer) if arm.output.citations else []
    cited, unknown = resolve_citations(labels, citation_map)

    return {
        "arm": arm.id,
        "question": question,
        "answer": answer,
        "refused": detect_refusal(answer),
        "refusal_mode": (
            "deterministic_empty_context"
            if not docs
            else ("prompt" if detect_refusal(answer) else "")
        ),
        "citations": citation_map,
        "cited": cited,
        "unknown_citations": unknown,
        "context": context,
        "retrieved_ids": [_doc_id(d) for d in docs],
        "candidate_ids": list(dict.fromkeys(candidate_ids)),
        "parent_ids": [],
        "queries": queries,
        "steps": steps,
        "usage": usage,
        "latency_sec": round(time.perf_counter() - start, 4),
        "error": "",
    }
