"""Experiment runner: 16 arms × cases × repeats, with retrieval reuse.

Retrieval is deterministic, so it runs once per (retrieval config, question)
and every repeat pays the measured retrieval latency. That keeps latency
honest while avoiding 3× the cross-encoder work on CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from arms.pipeline import RetrievalBundle, retrieve_for_arm, run_arm
from arms.schema import ARM_ORDER, ArmConfig, load_arms
from core.config import settings
from core.store import iter_collection_chunks, load_parents
from core.textnorm import normalize_text

from .case_loader import EvalCase, load_cases, resolve_evidence
from .judge import JUDGE_VERSION, judge_row
from core.console import enable_utf8
from .metrics import (
    bootstrap_mean_ci,
    mrr_at_k,
    ndcg_at_k,
    paired_comparisons,
    question_level_means,
    recall_at_k,
)

_write_lock = threading.Lock()


def _mean(values: Sequence[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return round(ordered[idx], 4)


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def retrieval_signature(arm: ArmConfig) -> tuple:
    r = arm.retrieval
    return (
        r.strategy,
        r.mode,
        r.k,
        r.fetch_k,
        r.hybrid,
        r.rerank,
        r.rerank_n,
        r.max_parents,
        arm.query.transform,
        arm.query.stepback_k,
    )


def build_chunk_indexes(arms: dict[str, ArmConfig]) -> dict[str, list[dict[str, Any]]]:
    strategies = sorted({arm.retrieval.strategy for arm in arms.values() if arm.uses_context})
    indexes: dict[str, list[dict[str, Any]]] = {}
    for strategy in strategies:
        chunks = list(iter_collection_chunks(strategy))
        if strategy == "parent_child":
            parents = load_parents()
            for cid, payload in parents.items():
                meta = payload.get("metadata") or {}
                chunks.append(
                    {
                        "chunk_id": cid,
                        "parent_id": cid,
                        "text": payload.get("text", ""),
                        "metadata": {**meta, "chunk_id": cid, "parent_id": cid},
                    }
                )
        indexes[strategy] = chunks
        print(f"[runner] index {strategy}: {len(chunks)} units")
    return indexes


def resolve_all(
    cases: Sequence[EvalCase], indexes: dict[str, list[dict[str, Any]]]
) -> dict[tuple[str, str], set[str]]:
    """(strategy, case_id) -> gold unit ids, resolved once per strategy."""
    out: dict[tuple[str, str], set[str]] = {}
    fallbacks = Counter()
    for strategy, chunks in indexes.items():
        parent_level = strategy == "parent_child"
        for case in cases:
            if not case.answerable:
                continue
            ids, method = resolve_evidence(
                case, chunks, ignore_page=parent_level, parent_level=parent_level
            )
            out[(strategy, case.id)] = ids
            if method == "page_fallback":
                fallbacks[case.id] += 1
    if fallbacks:
        print(f"[runner] WARNING evidence resolved by page fallback: {dict(fallbacks)}")
    return out


def gold_ids_for(arm: ArmConfig, case: EvalCase, resolved: dict) -> set[str]:
    return resolved.get((arm.retrieval.strategy, case.id), set())


def classify_error(
    arm: ArmConfig, row: dict[str, Any], gold: set[str], citation_arm: bool
) -> str:
    if row.get("error"):
        return "E6"
    answerable = bool(row.get("answerable"))
    refused = bool(row.get("refused"))
    if (answerable and refused) or (not answerable and not refused):
        return "E5"
    if not answerable:
        return ""
    if arm.uses_context and gold:
        candidates = set(row.get("candidate_ids") or [])
        retrieved = set(row.get("retrieved_ids") or [])
        if not gold & candidates:
            return "E1"
        if not gold & retrieved:
            return "E2"
    completeness = row.get("completeness")
    if isinstance(completeness, (int, float)) and completeness < 0.5:
        return "E3"
    precision = row.get("citation_precision")
    if citation_arm and isinstance(precision, (int, float)) and precision < 0.99:
        return "E4"
    return ""


def _precompute_retrieval(
    arms: dict[str, ArmConfig],
    cases: Sequence[EvalCase],
    needed: set[tuple[tuple, str]] | None = None,
) -> dict[tuple[tuple, str], RetrievalBundle]:
    jobs: dict[tuple[tuple, str], ArmConfig] = {}
    for arm in arms.values():
        if arm.agent.enabled or not arm.uses_context:
            continue
        signature = retrieval_signature(arm)
        for case in cases:
            key = (signature, case.id)
            if needed is not None and key not in needed:
                continue
            jobs.setdefault(key, arm)

    print(f"[runner] precomputing {len(jobs)} retrieval bundles")
    bundles: dict[tuple[tuple, str], RetrievalBundle] = {}
    started = time.perf_counter()
    # Sequential on purpose: the cross-encoder is CPU-bound and models are
    # process-wide, so parallel calls would only add memory pressure.
    for index, ((signature, case_id), arm) in enumerate(jobs.items(), start=1):
        case = next(c for c in cases if c.id == case_id)
        try:
            bundles[(signature, case_id)] = retrieve_for_arm(arm, case.question)
        except Exception as exc:
            print(f"[runner] retrieval failed {arm.id}/{case_id}: {exc}")
            bundles[(signature, case_id)] = RetrievalBundle(
                docs=[], candidate_ids=[], retrieved_ids=[], queries=[case.question]
            )
        if index % 20 == 0 or index == len(jobs):
            elapsed = time.perf_counter() - started
            print(f"  retrieval {index}/{len(jobs)} ({elapsed:.0f}s)")
    return bundles


def key_point_coverage(answer: str, key_points: Sequence[str]) -> float | None:
    if not key_points:
        return None
    text = normalize_text(answer)
    hits = sum(1 for kp in key_points if normalize_text(kp) in text)
    return round(hits / len(key_points), 4)


def _row_key(row: dict[str, Any]) -> tuple:
    return (row["arm"], row["case_id"], int(row["repeat"]))


def _build_row(
    arm: ArmConfig,
    case: EvalCase,
    repeat: int,
    run_id: str,
    result: dict[str, Any],
    judgment: dict[str, Any],
    gold: set[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "arm": arm.id,
        "arm_name": arm.name,
        "dimension": arm.dimension,
        "baseline": arm.baseline or "",
        "case_id": case.id,
        "base_id": case.base_id,
        "type": case.type,
        "language": case.language,
        "answerable": case.answerable,
        "question": case.question,
        "repeat": repeat,
        "answer": result.get("answer", ""),
        "refused": judgment.get("refused", False),
        "refused_det": judgment.get("refused_det", False),
        "completeness": judgment.get("completeness"),
        "relevance": judgment.get("relevance"),
        "relevance_binary": (
            None
            if judgment.get("relevance") is None
            else (1.0 if float(judgment["relevance"]) >= 0.5 else 0.0)
        ),
        "faithfulness": judgment.get("faithfulness"),
        "citation_precision": judgment.get("citation_precision"),
        "key_point_coverage": key_point_coverage(result.get("answer", ""), case.key_points),
        "judge_reason": judgment.get("reason", ""),
        "judge_error": judgment.get("judge_error", ""),
        "retrieved_ids": result.get("retrieved_ids", []),
        "candidate_ids": result.get("candidate_ids", []),
        "gold_ids": sorted(gold),
        "queries": result.get("queries", []),
        "steps": result.get("steps", []),
        "usage": result.get("usage", {}),
        "latency_sec": result.get("latency_sec", 0.0),
        "context": result.get("context", ""),
        "citations": [
            {
                "label": c.get("label"),
                "doc_title": c.get("doc_title"),
                "page": c.get("page"),
                "chunk_id": c.get("chunk_id"),
                "snippet": str(c.get("snippet", ""))[:300],
            }
            for c in result.get("citations", [])
        ],
        "unknown_citations": result.get("unknown_citations", []),
        "error": result.get("error", ""),
        "error_class": "",
    }


def run_experiment(
    *,
    arms: dict[str, ArmConfig],
    cases: Sequence[EvalCase],
    repeats: int = 3,
    workers: int = 8,
    out_dir: Path | None = None,
    resume: bool = True,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir or (settings.results_dir / run_id)
    out.mkdir(parents=True, exist_ok=True)

    indexes = build_chunk_indexes(arms)
    resolved = resolve_all(cases, indexes)

    rows: list[dict[str, Any]] = []
    raw_path = out / "raw.jsonl"
    done: set[tuple] = set()
    if resume and raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add(_row_key(row))
        print(f"[runner] resuming with {len(rows)} rows from {raw_path}")

    tasks = [
        (case, arm, repeat)
        for case in cases
        for arm in arms.values()
        for repeat in range(repeats)
        if (arm.id, case.id, repeat) not in done
    ]
    print(f"[runner] {len(tasks)} generation+judge tasks, workers={workers}")
    needed = {
        (retrieval_signature(arm), case.id)
        for case, arm, _repeat in tasks
        if not arm.agent.enabled
    }
    bundles = _precompute_retrieval(arms, cases, needed)

    completed = 0
    started = time.perf_counter()

    def work(task: tuple[EvalCase, ArmConfig, int]) -> dict[str, Any]:
        case, arm, repeat = task
        try:
            bundle = None
            if not arm.agent.enabled:
                bundle = bundles.get((retrieval_signature(arm), case.id))
            result = run_arm(arm, case.question, language=case.language, cached=bundle)
        except Exception as exc:
            result = {
                "arm": arm.id,
                "question": case.question,
                "answer": "",
                "refused": False,
                "citations": [],
                "context": "",
                "retrieved_ids": [],
                "candidate_ids": [],
                "queries": [],
                "usage": {},
                "latency_sec": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            judgment = judge_row(case, result, citation_required=arm.output.citations)
        except Exception as exc:
            judgment = {
                "completeness": None,
                "relevance": None,
                "faithfulness": None,
                "citation_precision": None,
                "refused": result.get("refused", False),
                "refused_det": result.get("refused", False),
                "reason": "",
                "judge_error": f"{type(exc).__name__}: {exc}",
            }
        gold = gold_ids_for(arm, case, resolved)
        row = _build_row(arm, case, repeat, run_id, result, judgment, gold)
        row["error_class"] = classify_error(arm, row, gold, arm.output.citations)
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            with _write_lock:
                rows.append(row)
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed += 1
            if completed % 25 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else 0
                print(
                    f"  {completed}/{len(tasks)} rows ({rate:.2f}/s, "
                    f"{elapsed / 60:.1f} min)"
                )

    # Sort deterministically so repeated runs diff cleanly.
    order = {arm_id: i for i, arm_id in enumerate(ARM_ORDER)}
    rows.sort(
        key=lambda r: (
            order.get(str(r["arm"]), 99),
            str(r["case_id"]),
            int(r["repeat"]),
        )
    )
    raw_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    write_outputs(out, arms, cases, rows)
    print(f"[runner] done in {(time.perf_counter() - started) / 60:.1f} min -> {out}")
    return out


def rejudge(
    out: Path, arms: dict[str, ArmConfig], cases: Sequence[EvalCase], workers: int = 8
) -> None:
    """Re-run only the judge over existing answers (e.g. after a prompt fix).

    Generation is unchanged; this keeps the expensive part of the raw data.
    """
    raw_path = out / "raw.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_case = {case.id: case for case in cases}

    def work(row: dict[str, Any]) -> dict[str, Any]:
        case = by_case.get(row["case_id"])
        arm = arms.get(row["arm"])
        if case is None or arm is None:
            return row
        judgment = judge_row(case, row, citation_required=arm.output.citations)
        row.update(
            {
                "refused": judgment.get("refused", False),
                "refused_det": judgment.get("refused_det", False),
                "completeness": judgment.get("completeness"),
                "relevance": judgment.get("relevance"),
                "faithfulness": judgment.get("faithfulness"),
                "citation_precision": judgment.get("citation_precision"),
                "judge_reason": judgment.get("reason", ""),
                "judge_error": judgment.get("judge_error", ""),
            }
        )
        gold = set(row.get("gold_ids") or [])
        row["error_class"] = classify_error(arm, row, gold, arm.output.citations)
        return row

    print(f"[rejudge] {len(rows)} rows -> {out}")
    updated: list[Any] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, row in enumerate(pool.map(work, rows), start=1):
            updated.append(row)
            if index % 50 == 0:
                print(f"  rejudged {index}/{len(rows)}")
    order = {arm_id: i for i, arm_id in enumerate(ARM_ORDER)}
    updated.sort(key=lambda r: (order.get(str(r["arm"]), 99), str(r["case_id"]), int(r["repeat"])))
    raw_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in updated) + "\n",
        encoding="utf-8",
    )
    write_outputs(out, arms, cases, updated)
    print("[rejudge] done")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_arm(
    arm: ArmConfig, rows: Sequence[dict[str, Any]], gold_map: dict[str, set[str]]
) -> dict[str, Any]:
    valid_rows = [r for r in rows if not r.get("error")]
    answerable_rows = [r for r in valid_rows if r["answerable"]]
    unanswerable_rows = [r for r in valid_rows if not r["answerable"]]

    def qmeans(key: str, exclude_refused: bool = False) -> list[float]:
        return question_level_means(answerable_rows, key, exclude_refused)

    completeness = qmeans("completeness")
    relevance = qmeans("relevance_binary") or qmeans("relevance")
    faithfulness = qmeans("faithfulness", exclude_refused=True)
    citation = qmeans("citation_precision", exclude_refused=True)
    ci_lo, ci_hi = bootstrap_mean_ci(completeness)

    latencies = [float(r.get("latency_sec", 0.0) or 0.0) for r in valid_rows]
    costs = [float((r.get("usage") or {}).get("cost_cny", 0.0) or 0.0) for r in rows]
    tokens = [int((r.get("usage") or {}).get("total_tokens", 0) or 0) for r in rows]

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    coverages: list[float] = []
    for row in answerable_rows:
        gold = set(row.get("gold_ids") or gold_map.get(row["case_id"], set()))
        if not gold:
            continue
        retrieved = [str(x) for x in row.get("retrieved_ids") or []]
        recalls.append(recall_at_k(gold, retrieved, 6))
        mrrs.append(mrr_at_k(gold, retrieved, 10))
        ndcgs.append(ndcg_at_k(gold, retrieved, 10))
        coverages.append(len(gold & set(retrieved)) / len(gold))

    errors = Counter(r.get("error_class") or "ok" for r in rows)
    # Metrics that do not apply to this arm are absent, not zero.
    faithfulness_value = _mean(faithfulness) if arm.uses_context else None
    citation_value = _mean(citation) if arm.output.citations else None
    recall_value = _mean(recalls) if (recalls and arm.uses_context) else None
    mrr_value = _mean(mrrs) if (mrrs and arm.uses_context) else None
    ndcg_value = _mean(ndcgs) if (ndcgs and arm.uses_context) else None
    coverage_value = _mean(coverages) if (coverages and arm.uses_context) else None
    return {
        "arm_id": arm.id,
        "arm_name": arm.name,
        "dimension": arm.dimension,
        "baseline": arm.baseline or "",
        "retrieval": arm.retrieval_label,
        "n_rows": len(rows),
        "n_questions": len({r["case_id"] for r in rows}),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "completeness": _mean(completeness),
        "completeness_ci_low": ci_lo,
        "completeness_ci_high": ci_hi,
        "relevance": _mean(relevance),
        "faithfulness": faithfulness_value,
        "citation_precision": citation_value,
        "key_points": _mean([v for v in qmeans("key_point_coverage")]),
        "refusal_correct": _mean([1.0 if r.get("refused") else 0.0 for r in unanswerable_rows]),
        "refusal_false": _mean([1.0 if r.get("refused") else 0.0 for r in answerable_rows]),
        "latency_p50": _median(latencies),
        "latency_p95": _p95(latencies),
        "cost_cny_per_q": _mean(costs),
        "tokens_per_q": _mean([float(t) for t in tokens]),
        "recall_at_6": recall_value,
        "mrr_at_10": mrr_value,
        "ndcg_at_10": ndcg_value,
        "evidence_coverage": coverage_value,
        "error_E1": errors.get("E1", 0),
        "error_E2": errors.get("E2", 0),
        "error_E3": errors.get("E3", 0),
        "error_E4": errors.get("E4", 0),
        "error_E5": errors.get("E5", 0),
        "error_E6": errors.get("E6", 0),
    }


def by_type_table(arms: dict[str, ArmConfig], rows: Sequence[dict[str, Any]]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["type"], row["language"])].append(row)
    out: list[dict] = []
    for (arm_id, case_type, language), items in sorted(grouped.items()):
        out.append(
            {
                "arm_id": arm_id,
                "type": case_type,
                "language": language,
                "n_rows": len(items),
                "completeness": _mean(question_level_means(items, "completeness")),
                "faithfulness": _mean(
                    question_level_means(items, "faithfulness", exclude_refused=True)
                ),
                "citation_precision": _mean(
                    question_level_means(items, "citation_precision", exclude_refused=True)
                ),
                "refusal_rate": _mean([1.0 if r.get("refused") else 0.0 for r in items]),
                "latency_p50": _median([r.get("latency_sec", 0.0) for r in items]),
            }
        )
    return out


def paired_table(arms: dict[str, ArmConfig], rows: Sequence[dict[str, Any]]) -> list[dict]:
    out: list[dict] = []
    metrics = ("completeness", "faithfulness", "citation_precision")
    for arm in arms.values():
        if not arm.baseline or arm.baseline == arm.id:
            continue
        for metric in metrics:
            out.extend(
                paired_comparisons(rows, arm.baseline, [arm.id], metric=metric)
            )
    return out


def by_language_table(arms: dict[str, ArmConfig], rows: Sequence[dict]) -> list[dict]:
    out: list[dict] = []
    for arm in arms.values():
        for language in sorted({r["language"] for r in rows}):
            items = [r for r in rows if r["arm"] == arm.id and r["language"] == language]
            if not items:
                continue
            answerable = [r for r in items if r["answerable"]]
            out.append(
                {
                    "arm_id": arm.id,
                    "language": language,
                    "n_questions": len({r["case_id"] for r in items}),
                    "completeness": _mean(question_level_means(answerable, "completeness")),
                    "faithfulness": _mean(
                        question_level_means(answerable, "faithfulness", exclude_refused=True)
                    ),
                    "citation_precision": _mean(
                        question_level_means(answerable, "citation_precision", exclude_refused=True)
                    ),
                    "refusal_rate": _mean([1.0 if r.get("refused") else 0.0 for r in items]),
                    "latency_p50": _median([r.get("latency_sec", 0.0) for r in items]),
                }
            )
    return out


def write_outputs(
    out: Path, arms: dict[str, ArmConfig], cases: Sequence[EvalCase], rows: Sequence[dict]
) -> None:
    # Main metrics are Chinese-only; language variants are reported separately.
    main_rows = [r for r in rows if r["language"] == "zh"]
    present = {str(r["arm"]) for r in rows}
    ran_arms = {arm_id: arm for arm_id, arm in arms.items() if arm_id in present}
    summary = [
        summarize_arm(arm, [r for r in main_rows if r["arm"] == arm.id], {})
        for arm in ran_arms.values()
    ]
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "by_type.csv", by_type_table(ran_arms, main_rows))
    _write_csv(out / "by_language.csv", by_language_table(ran_arms, rows))
    _write_csv(out / "paired.csv", paired_table(ran_arms, main_rows))

    error_rows = [
        {
            "arm": r["arm"],
            "case_id": r["case_id"],
            "type": r["type"],
            "repeat": r["repeat"],
            "error_class": r["error_class"],
            "refused": r["refused"],
            "completeness": r["completeness"],
            "citation_precision": r["citation_precision"],
            "reason": r["judge_reason"],
        }
        for r in rows
        if r.get("error_class") and r["error_class"] != "ok"
    ]
    _write_csv(out / "errors.csv", error_rows)

    meta = {
        "run_id": rows[0]["run_id"] if rows else "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "cases": len({r["case_id"] for r in rows}),
        "arms": list(arms),
        "model": settings.deepseek_model,
        "temperature": settings.temperature,
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "judge_model": settings.judge_model,
        "judge_version": JUDGE_VERSION,
        "questions_sha256": hashlib.sha256(
            settings.questions_path.read_bytes()
        ).hexdigest()
        if settings.questions_path.exists()
        else "",
        "arm_configs": {
            arm_id: {
                "dimension": arm.dimension,
                "baseline": arm.baseline,
                "retrieval": arm.retrieval.__dict__,
                "query": arm.query.__dict__,
                "prompt": arm.prompt.__dict__,
                "output": arm.output.__dict__,
                "agent": arm.agent.__dict__,
            }
            for arm_id, arm in arms.items()
        },
    }
    (out / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# RAG 对比实验运行报告", "", f"- Run: `{meta['run_id']}`", f"- Rows: {len(rows)}"]
    lines.append("")
    lines.append("| Arm | 完整性 | 相关性 | 忠实度 | 引用精度 | 正确拒答 | 错误拒答 | 延迟p50 | 延迟p95 | 成本/题 | 错误数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    def cell(value: Any, digits: int = 3) -> str:
        if value in (None, ""):
            return "—"
        return f"{float(value):.{digits}f}"

    for s in summary:
        lines.append(
            "| {arm} | {comp} | {rel} | {faith} | {cite} | {rc} | {rf} | "
            "{p50}s | {p95}s | ¥{cost} | {err} |".format(
                arm=s["arm_id"],
                comp=cell(s.get("completeness")),
                rel=cell(s.get("relevance")),
                faith=cell(s.get("faithfulness")),
                cite=cell(s.get("citation_precision")),
                rc=cell(s.get("refusal_correct")),
                rf=cell(s.get("refusal_false")),
                p50=cell(s.get("latency_p50"), 2),
                p95=cell(s.get("latency_p95"), 2),
                cost=cell(s.get("cost_cny_per_q"), 4),
                err=s.get("n_errors", 0),
            )
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[runner] wrote summary.csv / by_type.csv / paired.csv / errors.csv / run_meta.json")


def main() -> None:
    enable_utf8()
    ap = argparse.ArgumentParser(description="Run the 16-arm RAG comparison")
    ap.add_argument("--arms", nargs="+", default=None, help="subset of arm ids")
    ap.add_argument("--cases", nargs="+", default=None, help="subset of case ids")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--include-variants", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 cases × 1 repeat × 3 core arms")
    ap.add_argument("--limit", type=int, default=0, help="first N cases")
    ap.add_argument("--rejudge", action="store_true", help="re-run judge over an existing run")
    args = ap.parse_args()

    arms = load_arms(Path(__file__).resolve().parents[1] / "arms" / "configs")
    cases = load_cases(settings.questions_path, include_variants=args.include_variants)

    if args.cases:
        cases = [c for c in cases if c.id in set(args.cases)]
    if args.limit:
        cases = cases[: args.limit]
    if args.arms:
        missing = [a for a in args.arms if a not in arms]
        if missing:
            raise SystemExit(f"unknown arms: {missing}")
        arms = {a: arms[a] for a in args.arms}
    repeats = args.repeats
    workers = args.workers
    if args.smoke:
        cases = [c for c in cases if c.answerable][:1] + [
            c for c in cases if not c.answerable
        ][:1]
        arms = {a: arms[a] for a in ("A", "B", "C", "D") if a in arms}
        repeats = 1
        workers = 2

    if args.rejudge:
        if not args.out_dir:
            raise SystemExit("--rejudge needs --out-dir")
        rejudge(Path(args.out_dir), arms, cases, workers=workers)
        return

    out = run_experiment(
        arms=arms,
        cases=cases,
        repeats=repeats,
        workers=workers,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        resume=not args.no_resume,
    )
    print(out)


if __name__ == "__main__":
    main()
