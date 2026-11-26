from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.documents import Document

from core.citations import (
    detect_refusal,
    format_context,
    parse_citation_labels,
    refusal_text,
    resolve_citations,
)
from core.config import settings
from core.llm import complete
from core.store import RetrievalResult, expand_to_parents, retrieve
from prompts.templates import (
    COT_INSTRUCTION,
    SIMPLE_EXAMPLES,
    STEPBACK_SYSTEM,
    STEPBACK_USER,
    STRUCTURED_EXAMPLES,
    T1_CLOSED_BOOK_SYSTEM,
    T1_CLOSED_BOOK_USER,
    T1_SYSTEM,
    T1_USER,
    T2_NO_CONTEXT_SYSTEM,
    T2_SYSTEM,
    T2_USER,
    language_directive,
)

from .schema import ArmConfig


def empty_usage() -> dict[str, Any]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_cny": 0.0}


def add_usage(total: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(total)
    for key in ("input_tokens", "output_tokens", "total_tokens", "cost_cny"):
        out[key] = round(float(out.get(key, 0.0)) + float(extra.get(key, 0.0) or 0.0), 8)
    return out


def _doc_id(doc: Document) -> str:
    meta = doc.metadata or {}
    return str(meta.get("chunk_id") or meta.get("parent_id") or "")


def _dedupe(docs: Sequence[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        key = _doc_id(doc)
        if key and key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


@dataclass
class RetrievalBundle:
    docs: list[Document]
    candidate_ids: list[str]
    retrieved_ids: list[str]
    parent_ids: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=empty_usage)
    latency_sec: float = 0.0


def generate_stepback(question: str) -> tuple[str, dict[str, Any], float]:
    text, usage, latency = complete(
        [("system", STEPBACK_SYSTEM), ("human", STEPBACK_USER.format(question=question))]
    )
    abstract = (text or "").strip().splitlines()
    return (abstract[-1].strip() if abstract else question), usage, latency


def _retrieve_once(arm: ArmConfig, query: str, k: int) -> RetrievalResult:
    r = arm.retrieval
    return retrieve(
        strategy=r.strategy,
        query=query,
        mode=r.mode,
        k=k,
        fetch_k=max(r.fetch_k, k),
        use_hybrid=r.hybrid,
        use_rerank=r.rerank,
        rerank_n=r.rerank_n,
        rrf_k=settings.rrf_k,
    )


def retrieve_for_arm(arm: ArmConfig, question: str) -> RetrievalBundle:
    """Run the arm's retrieval chain once (deterministic, reused across repeats)."""
    start = time.perf_counter()
    usage = empty_usage()
    r = arm.retrieval

    if arm.query.transform == "stepback":
        abstract, step_usage, _ = generate_stepback(question)
        usage = add_usage(usage, step_usage)
        per_query = max(1, arm.query.stepback_k)
        first = _retrieve_once(arm, question, per_query)
        second = _retrieve_once(arm, abstract, per_query)
        docs = _dedupe([*first.docs, *second.docs])
        # Total context is held at k=6: dedupe first, then refill from the
        # original query's ranked candidates. This is what makes C-q attributable
        # to query construction rather than to more context.
        if len(docs) < r.k:
            for doc in [*first.candidates, *second.candidates]:
                if _doc_id(doc) in {_doc_id(d) for d in docs}:
                    continue
                docs.append(doc)
                if len(docs) >= r.k:
                    break
        docs = docs[: r.k]
        candidates = _dedupe([*first.candidates, *second.candidates])
        return RetrievalBundle(
            docs=docs,
            candidate_ids=[_doc_id(d) for d in candidates],
            retrieved_ids=[_doc_id(d) for d in docs],
            queries=[question, abstract],
            usage=usage,
            latency_sec=round(time.perf_counter() - start, 4),
        )

    k = r.rerank_n if r.max_parents else r.k
    result = _retrieve_once(arm, question, k)

    if r.max_parents:
        parents = expand_to_parents(result.docs, r.max_parents)
        return RetrievalBundle(
            docs=parents,
            candidate_ids=[_doc_id(d) for d in result.candidates],
            retrieved_ids=[_doc_id(d) for d in parents],
            parent_ids=[str((d.metadata or {}).get("parent_id", "")) for d in parents],
            queries=[question],
            usage=usage,
            latency_sec=round(time.perf_counter() - start, 4),
        )

    return RetrievalBundle(
        docs=result.docs,
        candidate_ids=[_doc_id(d) for d in result.candidates],
        retrieved_ids=[_doc_id(d) for d in result.docs],
        queries=[question],
        usage=usage,
        latency_sec=round(time.perf_counter() - start, 4),
    )


def _fewshot_messages(arm: ArmConfig, n: int) -> list[tuple[str, str]]:
    source = STRUCTURED_EXAMPLES if arm.prompt.template == "T2" else SIMPLE_EXAMPLES
    messages: list[tuple[str, str]] = []
    for example in source[: max(0, n)]:
        if arm.prompt.template == "T2":
            human = T2_USER.format(context=example["context"], question=example["question"])
        else:
            human = T1_USER.format(context=example["context"], question=example["question"])
        messages.append(("human", human))
        messages.append(("ai", example["answer"]))
    return messages


def build_messages(
    arm: ArmConfig, question: str, context: str, language: str = "zh"
) -> list[Any]:
    has_context = bool(context)
    cot = f"\n\n{COT_INSTRUCTION}" if arm.prompt.cot else ""

    if arm.prompt.template == "T2":
        system = (T2_SYSTEM if has_context else T2_NO_CONTEXT_SYSTEM) + cot
        user = (
            T2_USER.format(context=context, question=question)
            if has_context
            else f"问题：{question}\n\n回答："
        )
    else:
        system = (T1_SYSTEM if has_context else T1_CLOSED_BOOK_SYSTEM) + cot
        user = (
            T1_USER.format(context=context, question=question)
            if has_context
            else T1_CLOSED_BOOK_USER.format(question=question)
        )

    user = f"{user}\n\n{language_directive(language)}"

    messages: list[Any] = [("system", system)]
    if arm.prompt.fewshot:
        messages.extend(_fewshot_messages(arm, arm.prompt.fewshot))
    messages.append(("human", user))
    return messages


def run_arm(
    arm: ArmConfig,
    question: str,
    language: str = "zh",
    cached: RetrievalBundle | None = None,
) -> dict[str, Any]:
    """Run one arm on one question. Retrieval can be injected for repeats."""
    if arm.agent.enabled:
        from .agent import run_agent

        return run_agent(arm, question, language=language, cached=cached)

    start = time.perf_counter()
    if arm.uses_context:
        bundle = cached or retrieve_for_arm(arm, question)
    else:
        bundle = RetrievalBundle(
            docs=[], candidate_ids=[], retrieved_ids=[], queries=[question]
        )
    usage = dict(bundle.usage)
    # A cached bundle keeps its measured retrieval cost: repeats must not look
    # faster just because the retriever (especially the cross-encoder) was skipped.
    retrieval_latency = bundle.latency_sec if cached is not None else round(
        time.perf_counter() - start, 4
    )

    if not bundle.docs:
        # Deterministic refusal: never pay for a generation call when the
        # retriever returned nothing and the arm is allowed to abstain.
        if arm.output.refusal:
            return {
                "arm": arm.id,
                "question": question,
                "answer": refusal_text(language),
                "refused": True,
                "refusal_mode": "deterministic_empty_context",
                "citations": [],
                "context": "",
                "retrieved_ids": [],
                "candidate_ids": bundle.candidate_ids,
                "parent_ids": [],
                "queries": bundle.queries,
                "usage": usage,
                "latency_sec": round(retrieval_latency, 4),
                "error": "",
            }

    context, citation_map = format_context(bundle.docs)
    messages = build_messages(arm, question, context, language=language)
    answer, gen_usage, gen_latency = complete(messages)
    usage = add_usage(usage, gen_usage)

    labels = parse_citation_labels(answer) if arm.output.citations else []
    cited, unknown = resolve_citations(labels, citation_map)

    return {
        "arm": arm.id,
        "question": question,
        "answer": answer,
        "refused": detect_refusal(answer),
        "refusal_mode": "prompt" if detect_refusal(answer) else "",
        "citations": citation_map,
        "cited": cited,
        "unknown_citations": unknown,
        "context": context,
        "retrieved_ids": bundle.retrieved_ids,
        "candidate_ids": bundle.candidate_ids,
        "parent_ids": bundle.parent_ids,
        "queries": bundle.queries,
        "usage": usage,
        "latency_sec": round(retrieval_latency + gen_latency, 4),
        "error": "",
    }

