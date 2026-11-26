from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Sequence

from langchain_core.documents import Document

from .config import settings

_CITE_RE = re.compile(r"[\[［]\s*[Cc]\s*(\d+)\s*[\]］]")


@lru_cache(maxsize=1)
def _document_registry() -> dict[str, dict]:
    path = settings.corpus_dir / "documents.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _doi(meta: dict) -> str:
    if meta.get("doc_doi"):
        return str(meta["doc_doi"])
    entry = _document_registry().get(str(meta.get("doc") or ""), {})
    return str(entry.get("doi") or "")


def _page_label(meta: dict) -> str:
    start = meta.get("page_start", meta.get("page"))
    end = meta.get("page_end", start)
    if start is None:
        return "?"
    return f"{start}" if end in (None, start) else f"{start}-{end}"


def format_context(docs: Sequence[Document]) -> tuple[str, list[dict[str, Any]]]:
    """Render retrieved chunks as [C#] blocks and return the citation map.

    The label → source mapping is what the citation-precision judge checks and
    what the demo renders as source cards.
    """
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        label = f"C{index}"
        title = str(meta.get("doc_title") or meta.get("doc") or "unknown")
        page = _page_label(meta)
        heading = str(meta.get("heading_path") or "")
        header = f"[{label}] {title} | page={page}"
        if heading:
            header += f" | section={heading}"
        blocks.append(f"{header}\n{doc.page_content or ''}")
        citations.append(
            {
                "label": label,
                "doc": str(meta.get("doc") or ""),
                "doc_title": title,
                "page": page,
                "chunk_id": str(meta.get("chunk_id") or meta.get("parent_id") or ""),
                "parent_id": str(meta.get("parent_id") or ""),
                "heading_path": heading,
                "doi": _doi(meta),
                "snippet": str(meta.get("citation_snippet") or doc.page_content or "")[:600],
            }
        )
    return "\n\n".join(blocks), citations


def parse_citation_labels(answer: str) -> list[str]:
    seen: list[str] = []
    for match in _CITE_RE.finditer(answer or ""):
        label = f"C{int(match.group(1))}"
        if label not in seen:
            seen.append(label)
    return seen


def resolve_citations(
    labels: Sequence[str], citations: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    by_label = {c["label"]: c for c in citations}
    cited = [by_label[label] for label in labels if label in by_label]
    unknown = [label for label in labels if label not in by_label]
    return cited, unknown


def refusal_text(language: str = "zh") -> str:
    if language.startswith("de"):
        return "Mit den vorliegenden Unterlagen kann diese Frage nicht beantwortet werden."
    if language.startswith("en"):
        return "Based on the available documents, this question cannot be answered."
    return "根据现有资料无法回答。"


def detect_refusal(answer: str) -> bool:
    """Refusal = the answer offers no substantive answer and says so.

    A five-section answer may mention "无法回答" inside 【不确定项】 while still
    answering the question; only the opening of the answer decides.
    """
    text = (answer or "").strip()
    if not text:
        return True
    markers = (
        "根据现有资料无法回答",
        "无法回答",
        "资料不足",
        "insufficient_context",
        "insufficient context",
        "cannot be answered",
        "can not be answered",
        "nicht beantwortet werden",
        "keine ausreichenden informationen",
    )
    head = text[:300].lower()
    return any(marker in head for marker in markers)
