from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Sequence

from langchain_openai import ChatOpenAI

from .config import settings
from .pricing import estimate_cost


def _build_llm(model: str, temperature: float) -> ChatOpenAI:
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in .env")
    return ChatOpenAI(
        model=model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return _build_llm(settings.deepseek_model, settings.temperature)


@lru_cache(maxsize=1)
def get_judge_llm() -> ChatOpenAI:
    return _build_llm(settings.judge_model, settings.judge_temperature)


def _usage(response: Any) -> dict[str, Any]:
    meta = getattr(response, "usage_metadata", None) or {}
    inp = int(meta.get("input_tokens", 0) or 0)
    out = int(meta.get("output_tokens", 0) or 0)
    total = int(meta.get("total_tokens", inp + out) or (inp + out))
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
        "cost_cny": estimate_cost(inp, out),
    }


def complete(
    messages: Sequence[Any],
    *,
    judge: bool = False,
    temperature: float | None = None,
) -> tuple[str, dict[str, Any], float]:
    """Invoke the model once; returns (text, usage, latency_seconds)."""
    start = time.perf_counter()
    llm = get_judge_llm() if judge else get_llm()
    if temperature is not None:
        llm = llm.bind(temperature=temperature)
    response = llm.invoke(list(messages))
    content = getattr(response, "content", None)
    if not isinstance(content, str):
        content = str(response)
    return content, _usage(response), round(time.perf_counter() - start, 4)
