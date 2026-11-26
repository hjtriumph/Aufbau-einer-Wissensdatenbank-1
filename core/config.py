from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # generation (locked across all arms)
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    temperature: float
    max_tokens: int | None
    request_timeout: float
    max_retries: int

    # judge (separate call, temperature 0)
    judge_model: str
    judge_temperature: float

    # retrieval components (locked)
    embedding_model: str
    rerank_model: str
    rrf_k: int
    hybrid_fetch_k: int
    rerank_candidates: int
    k: int

    # chunking
    fixed_chunk_size: int
    fixed_chunk_overlap: int
    structure_chunk_size: int
    child_size: int
    parent_size: int
    max_parents: int

    # agent
    agent_max_steps: int
    agent_max_total_chunks: int

    # pricing (CNY per 1M tokens)
    price_input_per_m: float
    price_output_per_m: float

    # paths
    raw_dir: Path
    corpus_dir: Path
    chroma_dir: Path
    parsed_dir: Path
    parents_dir: Path
    cache_dir: Path
    results_dir: Path
    questions_path: Path
    variants_path: Path

    @property
    def has_api_key(self) -> bool:
        return bool(self.deepseek_api_key)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def load_settings() -> Settings:
    max_tokens_raw = os.getenv("MAX_TOKENS", "").strip()
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=_float("TEMPERATURE", 0.2),
        max_tokens=int(max_tokens_raw) if max_tokens_raw else None,
        request_timeout=_float("REQUEST_TIMEOUT", 120.0),
        max_retries=_int("MAX_RETRIES", 3),
        judge_model=os.getenv("JUDGE_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        judge_temperature=_float("JUDGE_TEMPERATURE", 0.0),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rrf_k=_int("RRF_K", 60),
        hybrid_fetch_k=_int("HYBRID_FETCH_K", 20),
        rerank_candidates=_int("RERANK_CANDIDATES", 20),
        k=_int("K", 6),
        fixed_chunk_size=_int("FIXED_CHUNK_SIZE", 1000),
        fixed_chunk_overlap=_int("FIXED_CHUNK_OVERLAP", 150),
        structure_chunk_size=_int("STRUCTURE_CHUNK_SIZE", 1200),
        child_size=_int("CHILD_SIZE", 250),
        parent_size=_int("PARENT_SIZE", 1800),
        max_parents=_int("MAX_PARENTS", 3),
        agent_max_steps=_int("AGENT_MAX_STEPS", 4),
        agent_max_total_chunks=_int("AGENT_MAX_TOTAL_CHUNKS", 18),
        price_input_per_m=_float("PRICE_INPUT_PER_M", 2.0),
        price_output_per_m=_float("PRICE_OUTPUT_PER_M", 8.0),
        raw_dir=_env_path("RAW_DIR", ROOT / "corpus" / "raw"),
        corpus_dir=_env_path("CORPUS_DIR", ROOT / "corpus"),
        chroma_dir=_env_path("CHROMA_DIR", ROOT / "storage" / "chroma"),
        parsed_dir=_env_path("PARSE_CACHE_DIR", ROOT / "storage" / "parsed"),
        parents_dir=_env_path("PARENTS_DIR", ROOT / "storage" / "parents"),
        cache_dir=_env_path("CACHE_DIR", ROOT / "storage" / "cache"),
        results_dir=_env_path("RESULTS_DIR", ROOT / "results"),
        questions_path=_env_path("QUESTIONS_PATH", ROOT / "corpus" / "questions.jsonl"),
        variants_path=_env_path("VARIANTS_PATH", ROOT / "corpus" / "variants.jsonl"),
    )


settings = load_settings()
