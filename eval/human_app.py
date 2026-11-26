"""Blind human-scoring web app.

Run:
  python -m uvicorn eval.human_app:APP --port 8001

The page never receives arm names or judge scores. Scores are saved to
results/<run>/human_scores.json and mirrored into human_check.csv so the CLI
(`python -m eval.human score`) keeps working.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.config import settings
from core.console import enable_utf8

from .case_loader import load_cases
from .human import score_rows

APP = FastAPI(title="Human calibration")
PAGE = Path(__file__).resolve().parent / "human.html"
_LOCK = threading.Lock()

FIELDNAMES = [
    "row_id",
    "question",
    "reference_answer",
    "key_points",
    "answer",
    "judge_completeness",
    "judge_relevance",
    "judge_refused",
    "human_completeness",
    "human_relevance",
    "human_refused",
    "notes",
]


def _run_dir() -> Path:
    override = os.getenv("HUMAN_RUN_DIR", "").strip()
    if override:
        return Path(override).resolve()
    candidates = [
        p for p in settings.results_dir.iterdir() if (p / "human_check.csv").exists()
    ]
    if not candidates:
        raise RuntimeError("no run with human_check.csv; run eval.human export first")
    return max(candidates, key=lambda p: (p / "human_check.csv").stat().st_mtime)


RUN_DIR = _run_dir()
CHECK_CSV = RUN_DIR / "human_check.csv"
KEY_CSV = RUN_DIR / "human_check_key.csv"
SCORES_JSON = RUN_DIR / "human_scores.json"
TRANSLATIONS_JSON = RUN_DIR / "human_translations.json"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_scores() -> dict[str, dict[str, Any]]:
    if not SCORES_JSON.exists():
        return {}
    try:
        return json.loads(SCORES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _has_score(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _merge() -> list[dict[str, Any]]:
    rows = _read_csv(CHECK_CSV)
    scores = _load_scores()
    for row in rows:
        saved = scores.get(str(row.get("row_id")), {})
        if saved:
            comp = saved.get("completeness")
            row["human_completeness"] = "" if comp is None else comp
            row["human_relevance"] = saved.get("relevance") if saved.get("relevance") is not None else ""
            row["human_refused"] = saved.get("refused") if saved.get("refused") is not None else ""
            row["notes"] = saved.get("notes", "")
    return rows


def _mirror_csv(rows: list[dict[str, Any]]) -> None:
    tmp = CHECK_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(CHECK_CSV)


def _progress(rows: list[dict[str, Any]]) -> dict[str, int]:
    done = sum(1 for r in rows if _has_score(r.get("human_completeness")))
    return {"done": done, "total": len(rows)}


def _translations() -> dict[str, dict]:
    if not TRANSLATIONS_JSON.exists():
        return {}
    try:
        return json.loads(TRANSLATIONS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _public_item(row: dict[str, Any], translations: dict[str, dict]) -> dict[str, Any]:
    # Deliberately omit judge_* fields: the annotator must score blind.
    scored = _has_score(row.get("human_completeness"))
    return {
        "row_id": int(row["row_id"]),
        "question": row.get("question", ""),
        "reference_answer": row.get("reference_answer", ""),
        "key_points": row.get("key_points", ""),
        "answer": row.get("answer", ""),
        "notes": row.get("notes", ""),
        "scored": scored,
        "human_completeness": row.get("human_completeness", ""),
        "human_relevance": row.get("human_relevance", ""),
        "human_refused": row.get("human_refused", ""),
        "i18n": translations.get(str(row["row_id"]), {}),
    }


class ScorePayload(BaseModel):
    row_id: int
    completeness: float | None = None
    relevance: int | None = None
    refused: int | None = None
    notes: str = Field(default="", max_length=2000)


@APP.get("/")
def index() -> FileResponse:
    return FileResponse(PAGE)


@APP.get("/api/human/meta")
def meta() -> dict[str, Any]:
    rows = _merge()
    return {"run": RUN_DIR.name, "progress": _progress(rows)}


@APP.get("/api/human/items")
def items() -> dict[str, Any]:
    rows = _merge()
    translations = _translations()
    return {
        "run": RUN_DIR.name,
        "progress": _progress(rows),
        "items": [_public_item(r, translations) for r in rows],
    }


@APP.post("/api/human/save")
def save(payload: ScorePayload) -> dict[str, Any]:
    if payload.completeness is not None and payload.completeness not in (0.0, 0.5, 1.0):
        raise HTTPException(status_code=400, detail="completeness must be 0, 0.5 or 1")
    if payload.relevance not in (None, 0, 1):
        raise HTTPException(status_code=400, detail="relevance must be 0 or 1")
    if payload.refused not in (None, 0, 1):
        raise HTTPException(status_code=400, detail="refused must be 0 or 1")

    with _LOCK:
        rows = _merge()
        if not any(int(r["row_id"]) == payload.row_id for r in rows):
            raise HTTPException(status_code=404, detail="unknown row_id")
        scores = _load_scores()
        scores[str(payload.row_id)] = {
            "completeness": payload.completeness,
            "relevance": payload.relevance,
            "refused": payload.refused,
            "notes": payload.notes,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _atomic_write(SCORES_JSON, json.dumps(scores, ensure_ascii=False, indent=2))
        rows = _merge()
        _mirror_csv(rows)
        return {"progress": _progress(rows), "saved": True}


def _per_group(
    rows: list[dict[str, Any]], key: dict[int, dict[str, str]], field: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if not _has_score(row.get("human_completeness")):
            continue
        try:
            human = float(row["human_completeness"])
            judge = float(row.get("judge_completeness") or 0)
        except (TypeError, ValueError):
            continue
        label = key.get(int(row["row_id"]), {}).get(field, "?")
        groups.setdefault(label, []).append((human, judge))
    out = []
    for label, pairs in sorted(groups.items()):
        out.append(
            {
                "label": label,
                "n": len(pairs),
                "human_mean": round(sum(p[0] for p in pairs) / len(pairs), 3),
                "judge_mean": round(sum(p[1] for p in pairs) / len(pairs), 3),
                "diff": round(
                    sum(p[0] for p in pairs) / len(pairs)
                    - sum(p[1] for p in pairs) / len(pairs),
                    3,
                ),
            }
        )
    return out


@APP.get("/api/human/result")
def result() -> dict[str, Any]:
    rows = _merge()
    progress = _progress(rows)
    if progress["done"] < progress["total"]:
        raise HTTPException(
            status_code=409,
            detail="score every row before revealing arm names and judge agreement",
        )

    key = {
        int(r["row_id"]): r
        for r in _read_csv(KEY_CSV)
        if str(r.get("row_id", "")).strip() != ""
    }
    type_by_case = {case.id: case.type for case in load_cases(settings.questions_path)}
    type_key = {
        row_id: {**data, "type": type_by_case.get(data.get("case_id", ""), "?")}
        for row_id, data in key.items()
    }

    return {
        "run": RUN_DIR.name,
        "progress": progress,
        "metrics": score_rows(rows),
        "by_arm": _per_group(rows, key, "arm"),
        "by_type": _per_group(rows, type_key, "type"),
    }


@APP.get("/api/human/export")
def export() -> FileResponse:
    rows = _merge()
    _mirror_csv(rows)
    return FileResponse(
        CHECK_CSV,
        media_type="text/csv",
        filename=f"human_check_{RUN_DIR.name}.csv",
    )


def main() -> None:
    enable_utf8()
    import uvicorn

    print(f"[human] run dir: {RUN_DIR}")
    uvicorn.run("eval.human_app:APP", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    main()
