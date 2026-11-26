from __future__ import annotations

from .config import settings


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Approximate CNY cost at the configured DeepSeek rates.

    Cache-hit discounts are ignored on purpose: the report must not understate
    the cost of a fresh run.
    """
    cost = (
        max(0, int(input_tokens)) / 1_000_000 * settings.price_input_per_m
        + max(0, int(output_tokens)) / 1_000_000 * settings.price_output_per_m
    )
    return round(cost, 8)
