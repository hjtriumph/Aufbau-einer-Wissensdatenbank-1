"""Demo API: C arm real-time QA + pre-cached A/B/C comparison.

Only the optimised arm is exposed for live questions. D and C-q are far slower
(multi-turn / extra LLM calls) and would turn a demo into a waiting room.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from arms.pipeline import RetrievalBundle, retrieve_for_arm, run_arm
from arms.schema import load_arm
from core.config import settings

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
APP = FastAPI(title="RAG Comparison Demo")

ARM_C = load_arm(ROOT / "arms" / "configs" / "C.yaml")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class WarmRequest(BaseModel):
    questions: list[str] = Field(default_factory=list, max_length=20)


# Retrieval is deterministic. Warming the sample questions makes the live demo
# respond quickly without changing the reported retrieval cost. One lock keeps
# the warm-up thread and a live request from touching Chroma/torch at the same
# time; retrieval is CPU-bound, so serialising costs nothing.
_BUNDLES: dict[str, RetrievalBundle] = {}
_BUNDLE_LOCK = threading.RLock()
_WARM_POOL = ThreadPoolExecutor(max_workers=1)


def _bundle(question: str) -> RetrievalBundle:
    with _BUNDLE_LOCK:
        if question in _BUNDLES:
            return _BUNDLES[question]
        bundle = retrieve_for_arm(ARM_C, question)
        _BUNDLES[question] = bundle
        return bundle


def _latest_run() -> Path | None:
    runs = [
        p
        for p in settings.results_dir.iterdir()
        if p.is_dir() and (p / "raw.jsonl").exists()
    ]
    if not runs:
        return None
    return max(runs, key=lambda p: (p / "raw.jsonl").stat().st_mtime)


@APP.get("/")
def index() -> FileResponse:
    return FileResponse(DEMO_DIR / "index.html")


@APP.get("/api/meta")
def meta() -> dict[str, Any]:
    run = _latest_run()
    return {
        "model": settings.deepseek_model,
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "arm": ARM_C.name,
        "run": run.name if run else "",
    }


@APP.post("/api/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty question")
    # Language follows the question; the prompt enforces the same rule.
    language = "de" if _looks_german(question) else ("en" if _looks_english(question) else "zh")
    started = time.perf_counter()
    with _BUNDLE_LOCK:
        was_cached = question in _BUNDLES
    bundle = _bundle(question)
    result = run_arm(ARM_C, question, language=language, cached=bundle)
    return {
        "question": question,
        "answer": result["answer"],
        "refused": result["refused"],
        "citations": [
            {
                "label": c["label"],
                "doc_title": c["doc_title"],
                "page": c["page"],
                "heading_path": c["heading_path"],
                "snippet": c["snippet"][:500],
            }
            for c in result["citations"]
        ],
        "latency_sec": result["latency_sec"],
        "wall_latency_sec": round(time.perf_counter() - started, 3),
        "retrieval_cached": was_cached,
        "usage": result["usage"],
        "queries": result.get("queries", []),
    }


@APP.post("/api/warm")
def warm(request: WarmRequest) -> dict[str, Any]:
    questions = [q.strip() for q in request.questions if q.strip()][:20]
    for question in questions:
        _WARM_POOL.submit(_bundle, question)
    return {"queued": len(questions)}


def _looks_german(text: str) -> bool:
    markers = ("der ", "die ", "das ", "und ", "wie ", "welche", "ist ", "sind ", "ä", "ö", "ü", "ß")
    low = text.lower()
    return sum(1 for m in markers if m in low) >= 2


def _looks_english(text: str) -> bool:
    markers = ("the ", "what ", "how ", "which ", "does ", "is ", "are ", "of ")
    low = text.lower()
    return sum(1 for m in markers if m in low) >= 2


def _comparison_translations(run: Path, lang: str) -> dict[str, dict]:
    if lang not in ("en", "de"):
        return {}
    path = run / "comparison_translations.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {case_id: data.get(lang, {}) for case_id, data in payload.items()}


@APP.get("/api/comparison")
def comparison(lang: str = "zh") -> dict[str, Any]:
    run = _latest_run()
    if not run:
        return {"run": "", "cases": [], "lang": lang, "translated": False}
    translations = _comparison_translations(run, lang)

    wanted_types = ("numeric", "multi_hop", "unanswerable")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with (run / "raw.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("repeat") != 0 or row.get("language") != "zh":
                continue
            if row.get("arm") not in ("A", "B", "C"):
                continue
            rows[(row["case_id"], row["arm"])] = row

    cases = []
    for case_type in wanted_types:
        case_ids = sorted({cid for (cid, arm) in rows if rows[(cid, arm)]["type"] == case_type})
        for case_id in case_ids[:1]:
            entry: dict[str, Any] = {"case_id": case_id, "type": case_type}
            translation = translations.get(case_id, {})
            translated = bool(translation)
            for arm in ("A", "B", "C"):
                row = rows.get((case_id, arm))
                if not row:
                    continue
                if "question" not in entry:
                    entry["question"] = translation.get("question") or row["question"]
                entry[arm] = {
                    "answer": row.get("answer", ""),
                    "refused": row.get("refused", False),
                    "completeness": row.get("completeness"),
                    "citation_precision": row.get("citation_precision"),
                    "citations": row.get("citations", [])[:6],
                }
                if translation.get(arm):
                    entry[arm]["answer"] = translation[arm]
            entry["translated"] = translated
            cases.append(entry)
    return {
        "run": run.name,
        "cases": cases,
        "lang": lang,
        "translated": any(c.get("translated") for c in cases),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("demo.app:APP", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
