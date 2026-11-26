"""Fill translation entries that failed as a whole payload (long answers).

Strategy: translate field by field; split an over-long answer at paragraph
boundaries. Any field that still fails is left out, and the UI falls back to the
Chinese original with a visible notice.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.translate_content import LANGS, _valid, translate

RUN = Path("results/full_gpu")
HUMAN = RUN / "human_translations.json"
COMPARE = RUN / "comparison_translations.json"


def cjk(text: str) -> int:
    return len(re.findall(r"[一-鿿]", text))


def translate_field(name: str, value: str, lang: str) -> str:
    try:
        return translate({name: value}, lang)[name]
    except Exception:
        pass
    if name != "answer" or len(value) < 1200:
        raise RuntimeError(f"field {name} failed")
    parts = [p for p in value.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) + 2 > 1800:
            chunks.append(current)
            current = part
        else:
            current = f"{current}\n\n{part}".strip() if current else part
    if current:
        chunks.append(current)
    out = []
    for chunk in chunks:
        out.append(translate({"answer": chunk}, lang)["answer"])
    return "\n\n".join(out)


def fix_human() -> None:
    data = json.loads(HUMAN.read_text(encoding="utf-8"))
    for row_id, langs in data.items():
        for lang in LANGS:
            if lang in langs and _valid(langs[lang], lang):
                continue
            source = langs.get("_source")
            # Recover the Chinese source from the CSV if needed.
            print(f"[fix] human {row_id} {lang}")
            import csv

            rows = {
                str(r["row_id"]): r
                for r in csv.DictReader(
                    (RUN / "human_check.csv").open(encoding="utf-8-sig")
                )
            }
            row = rows[row_id]
            payload = {
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "key_points": row["key_points"],
                "answer": row["answer"],
            }
            fixed = {}
            for field, value in payload.items():
                try:
                    fixed[field] = translate_field(field, value, lang)
                except Exception as exc:
                    print(f"  {field} failed: {exc}")
            if all(k in fixed for k in payload) and _valid(fixed, lang):
                data[row_id][lang] = fixed
                HUMAN.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                data[row_id].pop(lang, None)
                HUMAN.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fix_compare() -> None:
    data = json.loads(COMPARE.read_text(encoding="utf-8"))
    # Rebuild the Chinese source from raw.jsonl.
    rows = [
        json.loads(line)
        for line in (RUN / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source: dict[str, dict] = {}
    for row in rows:
        if row.get("repeat") == 0 and row.get("language") == "zh" and row.get("arm") in ("A", "B", "C"):
            source.setdefault(row["case_id"], {"question": row["question"]})
            source[row["case_id"]][row["arm"]] = row.get("answer", "")
    for case_id, langs in data.items():
        for lang in LANGS:
            if lang in langs and _valid(langs[lang], lang):
                continue
            if case_id not in source:
                continue
            print(f"[fix] compare {case_id} {lang}")
            payload = {"question": source[case_id]["question"]}
            for arm in ("A", "B", "C"):
                if arm in source[case_id]:
                    payload[arm] = source[case_id][arm]
            fixed = {}
            for field, value in payload.items():
                try:
                    fixed[field] = translate_field(field, value, lang)
                except Exception as exc:
                    print(f"  {field} failed: {exc}")
            if all(k in fixed for k in payload) and _valid(fixed, lang):
                data[case_id][lang] = fixed
            else:
                data[case_id].pop(lang, None)
            COMPARE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    fix_human()
    fix_compare()
    print("done")
