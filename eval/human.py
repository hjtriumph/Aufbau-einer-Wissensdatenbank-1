"""Human spot-check: export 25 rows, then compute agreement and weighted kappa.

Usage:
  python -m eval.human export --run results/YYYYMMDD-HHMMSS --n 25
  python -m eval.human score  --csv results/.../human_check.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

from core.config import settings

from .case_loader import load_cases
from .metrics import weighted_kappa

LABELS = (0.0, 0.5, 1.0)
BINARY = (0.0, 1.0)


def export_rows(run_dir: Path, n: int = 25, seed: int = 42) -> Path:
    raw_path = run_dir / "raw.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = [
        r
        for r in rows
        if r.get("repeat") == 0
        and r.get("language") == "zh"
        and r.get("answerable")
        and isinstance(r.get("completeness"), (int, float))
    ]
    rng = random.Random(seed)
    sample = rng.sample(candidates, k=min(n, len(candidates)))
    rng.shuffle(sample)

    # The scoring sheet is blind to arm: knowing the arm would bias the
    # annotator toward the expected winner. The mapping stays in a side file.
    cases = {case.id: case for case in load_cases(settings.questions_path)}
    path = run_dir / "human_check.csv"
    key_path = run_dir / "human_check_key.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f, key_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as kf:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        key_writer = csv.DictWriter(
            kf, fieldnames=["row_id", "arm", "case_id", "repeat"]
        )
        writer.writeheader()
        key_writer.writeheader()
        for index, row in enumerate(sample):
            case = cases.get(str(row["case_id"]))
            reference = case.reference_answer if case else ""
            key_points = "；".join(case.key_points) if case else ""
            writer.writerow(
                {
                    "row_id": index,
                    "question": row["question"],
                    "reference_answer": reference,
                    "key_points": key_points,
                    "answer": row.get("answer", ""),
                    "judge_completeness": row.get("completeness"),
                    "judge_relevance": row.get("relevance"),
                    "judge_refused": 1 if row.get("refused") else 0,
                    "human_completeness": "",
                    "human_relevance": "",
                    "human_refused": "",
                    "notes": "",
                }
            )
            key_writer.writerow(
                {
                    "row_id": index,
                    "arm": row["arm"],
                    "case_id": row["case_id"],
                    "repeat": row.get("repeat"),
                }
            )
    print(f"[human] exported {len(sample)} rows -> {path}")
    print(f"[human] blind key (do not open while scoring) -> {key_path}")
    print("[human] fill human_completeness (0 / 0.5 / 1), human_relevance (0 / 1), human_refused (0 / 1)")
    return path


def _score(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return min(LABELS, key=lambda x: abs(x - number))


def _binary(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return 1.0 if number >= 0.5 else 0.0


def score_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def collect(human_key: str, judge_key: str, parse) -> list[tuple[float, float]]:
        pairs: list[tuple[float, float]] = []
        for row in rows:
            human = parse(row.get(human_key))
            judge = parse(row.get(judge_key))
            if human is None or judge is None:
                continue
            pairs.append((human, judge))
        return pairs

    def agreement(
        pairs: Sequence[tuple[float, float]], labels: Sequence[float]
    ) -> dict[str, Any]:
        if not pairs:
            return {"n": 0}
        exact = sum(1 for a, b in pairs if abs(a - b) < 1e-9)
        within = sum(1 for a, b in pairs if abs(a - b) <= 0.25 + 1e-9)
        return {
            "n": len(pairs),
            "exact_agreement": round(exact / len(pairs), 4),
            "within_025": round(within / len(pairs), 4),
            "weighted_kappa": weighted_kappa(pairs, list(labels)),
        }

    completeness = collect("human_completeness", "judge_completeness", _score)
    relevance = collect("human_relevance", "judge_relevance", _binary)
    refusal = collect("human_refused", "judge_refused", _binary)

    result = {
        "completeness": agreement(completeness, LABELS),
        "relevance": agreement(relevance, BINARY),
        "refusal": agreement(refusal, BINARY),
    }
    return result


def score_csv(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    result = score_rows(rows)
    if not result["completeness"]["n"]:
        raise SystemExit("no completed human_completeness values found")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--run", required=True)
    exp.add_argument("--n", type=int, default=25)
    exp.add_argument("--seed", type=int, default=42)
    score = sub.add_parser("score")
    score.add_argument("--csv", required=True)
    args = ap.parse_args()

    if args.command == "export":
        export_rows(Path(args.run), n=args.n, seed=args.seed)
    else:
        score_csv(Path(args.csv))


if __name__ == "__main__":
    main()
