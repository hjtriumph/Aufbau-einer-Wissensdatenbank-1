"""Pre-translate fixed UI content (human-scoring rows, comparison answers).

Machine translation is a reading aid only: the Chinese benchmark stays
authoritative. Results are cached per row/language so this script is resumable.

  python scripts/translate_content.py --run results/full_gpu
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.console import enable_utf8
from core.llm import complete

LANGS = {"en": "English", "de": "German"}
_WRITE_LOCK = threading.Lock()


def _save(path: Path, data: dict) -> None:
    with _WRITE_LOCK:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

SYSTEM = """You are a professional translator for HVAC, building-energy and indoor-air research.
Translate every string value in the input JSON into {language}.
Rules:
- Preserve all numbers, units, percentages and ranges exactly.
- Preserve citation labels such as [C1] exactly and in place.
- Preserve paper titles and author names in their original form when they function as identifiers.
- Keep the original structure: the five-section answer must keep its five sections.
- Do not summarise, add, or omit content.
- Do not answer the question; only translate the text.
- Every output string MUST be written in {language}. Chinese characters are only allowed inside a paper title that has no official translation.
- Return JSON only, with exactly the same keys and shapes as the input."""

USER = "Input JSON:\n{payload}"


def _extract_json(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON in translation: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[一-鿿]", text))


def _valid(data: dict, lang: str) -> bool:
    joined = " ".join(str(v) for v in data.values())
    # A paper title may legitimately stay Chinese; a whole untranslated field
    # is hundreds of CJK characters and must be rejected.
    return _cjk_count(joined) <= 20


def translate(payload: dict, lang: str) -> dict:
    last_error = ""
    for attempt in range(3):
        text, _usage, _latency = complete(
            [
                ("system", SYSTEM.format(language=LANGS[lang])),
                ("human", USER.format(payload=json.dumps(payload, ensure_ascii=False))),
            ],
            temperature=0.0,
        )
        try:
            data = _extract_json(text)
            for key in payload:
                if key not in data:
                    raise ValueError(f"missing key {key}")
            if _valid(data, lang):
                return data
            last_error = f"output not in {lang} (attempt {attempt + 1})"
        except Exception as exc:
            last_error = str(exc)
    raise ValueError(last_error)


def human_items(run_dir: Path) -> tuple[Path, list[dict]]:
    csv_path = run_dir / "human_check.csv"
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return run_dir / "human_translations.json", rows


def comparison_items(run_dir: Path) -> tuple[Path, dict[str, dict]]:
    rows = [
        json.loads(line)
        for line in (run_dir / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = ("numeric", "multi_hop", "unanswerable")
    by_case: dict[str, dict] = {}
    for row in rows:
        if row.get("repeat") != 0 or row.get("language") != "zh":
            continue
        if row.get("arm") not in ("A", "B", "C"):
            continue
        if row["type"] not in wanted:
            continue
        by_case.setdefault(row["case_id"], {"type": row["type"], "question": row["question"], "answers": {}})
        by_case[row["case_id"]]["answers"][row["arm"]] = row.get("answer", "")
    # Keep the same first-per-type selection the demo uses.
    chosen: dict[str, dict] = {}
    for case_type in wanted:
        for case_id, data in sorted(by_case.items()):
            if data["type"] == case_type:
                chosen[case_id] = data
                break
    return run_dir / "comparison_translations.json", chosen


def main() -> None:
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/full_gpu")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run_dir = Path(args.run).resolve()

    human_path, human_rows = human_items(run_dir)
    cache: dict[str, dict] = {}
    if human_path.exists():
        cache = json.loads(human_path.read_text(encoding="utf-8"))

    jobs = []
    for row in human_rows:
        row_id = str(row["row_id"])
        cache.setdefault(row_id, {})
        for lang in LANGS:
            if lang in cache[row_id] and not _valid(cache[row_id][lang], lang):
                print(f"[translate] dropping invalid {lang} row {row_id}")
                cache[row_id].pop(lang)
            if lang in cache[row_id]:
                continue
            payload = {
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "key_points": row["key_points"],
                "answer": row["answer"],
            }
            jobs.append(("human", row_id, lang, payload))

    comp_path, comp = comparison_items(run_dir)
    comp_cache: dict[str, dict] = {}
    if comp_path.exists():
        comp_cache = json.loads(comp_path.read_text(encoding="utf-8"))
    for case_id, data in comp.items():
        comp_cache.setdefault(case_id, {})
        for lang in LANGS:
            if lang in comp_cache[case_id] and not _valid(comp_cache[case_id][lang], lang):
                print(f"[translate] dropping invalid {lang} case {case_id}")
                comp_cache[case_id].pop(lang)
            if lang in comp_cache[case_id]:
                continue
            payload = {"question": data["question"], **data["answers"]}
            jobs.append(("compare", case_id, lang, payload))

    _save(human_path, cache)
    _save(comp_path, comp_cache)
    print(f"[translate] {len(jobs)} jobs (human {len(human_rows)} rows, compare {len(comp)} cases)")

    def work(job):
        kind, key, lang, payload = job
        return kind, key, lang, translate(payload, lang)

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                kind, key, lang, data = future.result()
            except Exception as exc:
                print(f"  FAILED: {exc}")
                continue
            if kind == "human":
                cache[key][lang] = data
                _save(human_path, cache)
            else:
                comp_cache[key][lang] = data
                _save(comp_path, comp_cache)
            completed += 1
            if completed % 5 == 0 or completed == len(jobs):
                print(f"  {completed}/{len(jobs)}")

    print(f"[translate] human -> {human_path}")
    print(f"[translate] compare -> {comp_path}")


if __name__ == "__main__":
    main()
