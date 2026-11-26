from __future__ import annotations

import math
import random
import statistics
from typing import Sequence


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return round(hits / len(relevant), 4)


def precision_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    top = list(retrieved[:k])
    if not top:
        return 0.0
    return round(len(set(top) & relevant) / len(top), 4)


def mrr_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
        if doc in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return round(dcg / ideal, 4) if ideal else 0.0


def _bootstrap_means(values: Sequence[float], n_boot: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    return [statistics.mean(rng.choices(values, k=n)) for _ in range(n_boot)]


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    means = sorted(_bootstrap_means(values, n_boot, seed))
    lo = max(0, int((alpha / 2) * n_boot))
    hi = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (round(means[lo], 4), round(means[hi], 4))


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 42
) -> dict[str, float | int]:
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and equal length")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    means = sorted(_bootstrap_means(diffs, n_boot, seed))
    lo = max(0, int(0.025 * n_boot))
    hi = min(n_boot - 1, int(0.975 * n_boot))
    p_left = sum(1 for m in means if m <= 0) / n_boot
    p_right = sum(1 for m in means if m >= 0) / n_boot
    return {
        "mean_diff": round(statistics.mean(diffs), 4),
        "ci_low": round(means[lo], 4),
        "ci_high": round(means[hi], 4),
        "p_value": round(min(1.0, 2 * min(p_left, p_right)), 4),
        "n": len(diffs),
    }


def question_level_means(
    rows: Sequence[dict], metric: str, exclude_refused: bool = False
) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("error") or not row.get("answerable", True):
            continue
        if exclude_refused and row.get("refused"):
            continue
        value = row.get(metric)
        if isinstance(value, (int, float)):
            grouped.setdefault(str(row.get("case_id")), []).append(float(value))
    return [statistics.mean(v) for v in grouped.values()]


def _question_means(rows: Sequence[dict], arm: str, metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("arm") != arm or not row.get("answerable", True):
            continue
        value = row.get(metric)
        if isinstance(value, (int, float)):
            grouped.setdefault(str(row.get("case_id")), []).append(float(value))
    return {case_id: statistics.mean(values) for case_id, values in grouped.items()}


def paired_comparisons(
    rows: Sequence[dict],
    baseline: str,
    compares: Sequence[str],
    metric: str = "completeness",
    n_boot: int = 2000,
    seed: int = 42,
) -> list[dict]:
    base = _question_means(rows, baseline, metric)
    out: list[dict] = []
    for compare in compares:
        other = _question_means(rows, compare, metric)
        common = sorted(set(base) & set(other))
        if not common:
            continue
        stats = paired_bootstrap_diff(
            [other[c] for c in common],
            [base[c] for c in common],
            n_boot=n_boot,
            seed=seed,
        )
        stats.update({"baseline": baseline, "compare": compare, "metric": metric})
        out.append(stats)
    return out


def weighted_kappa(pairs: Sequence[tuple[float, float]], labels: Sequence[float]) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal judgments."""
    if not pairs:
        return 0.0
    index = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0.0] * n for _ in range(n)]
    for a, b in pairs:
        matrix[index[a]][index[b]] += 1
    total = len(pairs)

    observed = 0.0
    for i in range(n):
        for j in range(n):
            observed += matrix[i][j] * ((i - j) ** 2) / ((n - 1) ** 2)
    observed /= total

    row = [sum(matrix[i]) / total for i in range(n)]
    col = [sum(matrix[i][j] for i in range(n)) / total for j in range(n)]
    expected = sum(
        row[i] * col[j] * ((i - j) ** 2) / ((n - 1) ** 2) for i in range(n) for j in range(n)
    )
    if expected == 0:
        return 1.0
    return round(1 - observed / expected, 4)
