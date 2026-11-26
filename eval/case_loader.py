from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.textnorm import contains_quote

CASE_TYPES = ("numeric", "single_hop", "multi_hop", "method", "unanswerable")
LANGUAGES = ("zh", "en", "de")


@dataclass
class EvalCase:
    id: str
    question: str
    type: str
    reference_answer: str
    key_points: list[str] = field(default_factory=list)
    gold_evidence: list[dict[str, Any]] = field(default_factory=list)
    answerable: bool = True
    language: str = "zh"
    base_id: str = ""
    notes: str = ""


def _require(raw: dict[str, Any], key: str, case_id: str) -> Any:
    if key not in raw:
        raise ValueError(f"case {case_id}: missing {key}")
    return raw[key]


def load_cases(path: str | Path, include_variants: bool = False) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()

    paths = [Path(path)]
    if include_variants:
        variants = Path(path).with_name("variants.jsonl")
        if variants.exists():
            paths.append(variants)

    for case_path in paths:
        if not case_path.exists():
            continue
        for line_no, line in enumerate(case_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            case_id = str(_require(raw, "id", f"line {line_no}"))
            if case_id in seen:
                raise ValueError(f"duplicate case id: {case_id}")
            seen.add(case_id)

            case_type = str(_require(raw, "type", case_id))
            if case_type not in CASE_TYPES:
                raise ValueError(f"case {case_id}: bad type {case_type}")

            language = str(raw.get("language", "zh"))
            if language not in LANGUAGES:
                raise ValueError(f"case {case_id}: bad language {language}")

            answerable = bool(raw.get("answerable", case_type != "unanswerable"))
            evidence = raw.get("gold_evidence", [])
            if not isinstance(evidence, list):
                raise ValueError(f"case {case_id}: gold_evidence must be a list")
            if answerable and not evidence:
                raise ValueError(f"case {case_id}: answerable cases need gold evidence")
            if not answerable and evidence:
                raise ValueError(f"case {case_id}: unanswerable cases must not carry evidence")
            for ev in evidence:
                for key in ("doc", "page", "quote"):
                    if ev.get(key) in (None, ""):
                        raise ValueError(f"case {case_id}: evidence missing {key}")

            cases.append(
                EvalCase(
                    id=case_id,
                    question=str(_require(raw, "question", case_id)),
                    type=case_type,
                    reference_answer=str(_require(raw, "reference_answer", case_id)),
                    key_points=[str(x) for x in raw.get("key_points", [])],
                    gold_evidence=evidence,
                    answerable=answerable,
                    language=language,
                    base_id=str(raw.get("base_id", "")),
                    notes=str(raw.get("notes", "")),
                )
            )
    return cases


def _page_hit(metadata: dict, page: int, ignore_page: bool) -> bool:
    if ignore_page:
        return True
    start = metadata.get("page_start", metadata.get("page"))
    end = metadata.get("page_end", start)
    if start is None:
        return True
    try:
        return int(start) <= int(page) <= int(end)
    except (TypeError, ValueError):
        return True


def resolve_evidence(
    case: EvalCase,
    chunk_index: list[dict[str, Any]],
    *,
    ignore_page: bool = False,
    parent_level: bool = False,
) -> tuple[set[str], str]:
    """Map (doc, page, quote) to current chunk ids.

    Chunks differ per strategy, so evidence is resolved against the index the
    arm actually used. A page-scoped fallback is reported so a bad quote can
    never silently inflate the relevant set without leaving a trace.
    """
    matched: set[str] = set()
    used_fallback = False

    for ev in case.gold_evidence:
        doc = str(ev["doc"])
        page = int(ev["page"])
        quote = str(ev["quote"])
        id_key = "parent_id" if parent_level else "chunk_id"

        scoped = []
        for chunk in chunk_index:
            meta = chunk["metadata"]
            if str(meta.get("doc", "")) != doc:
                continue
            if not _page_hit(meta, page, ignore_page):
                continue
            scoped.append(chunk)

        hits = [
            str(c["metadata"][id_key])
            for c in scoped
            if c["metadata"].get(id_key) and contains_quote(c["text"], quote)
        ]
        if not hits:
            # Try sentence fragments; PDF extraction differences often break a
            # long quote at one point while the rest stays intact.
            fragments = [
                part.strip()
                for part in quote.replace("\n", " ").split(". ")
                if len(part.strip()) >= 30
            ]
            hits = [
                str(c["metadata"][id_key])
                for c in scoped
                if c["metadata"].get(id_key)
                and any(contains_quote(c["text"], fragment) for fragment in fragments)
            ]
        if not hits:
            hits = [
                str(c["metadata"][id_key])
                for c in scoped
                if c["metadata"].get(id_key)
            ]
            used_fallback = True
        matched.update(hits)

    return matched, ("page_fallback" if used_fallback else "quote")
