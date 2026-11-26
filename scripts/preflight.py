"""Pre-flight checks: env, indexes, evidence resolution, retrieval sanity.

Runs without any LLM generation call, so a broken index or an unresolvable
gold quote is caught before spending API budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import settings
from core.store import collection_count, dense_search, load_parents
from eval.case_loader import load_cases, resolve_evidence
from eval.runner import build_chunk_indexes, resolve_all
from arms.schema import load_arms
from core.console import enable_utf8


def main() -> None:
    enable_utf8()
    print(f"API key set: {settings.has_api_key}")
    print(f"embedding model: {settings.embedding_model}")
    print(f"rerank model: {settings.rerank_model}")
    print(f"chroma dir: {settings.chroma_dir}")
    for strategy in ("fixed", "structure", "parent_child"):
        print(f"collection {strategy}: {collection_count(strategy)} units")
    print(f"parents: {len(load_parents())}")

    arms = load_arms(Path(__file__).resolve().parents[1] / "arms" / "configs")
    indexes = build_chunk_indexes(arms)
    cases = load_cases(settings.questions_path)
    print(f"cases: {len(cases)} ({sum(1 for c in cases if c.answerable)} answerable)")

    resolved = resolve_all(cases, indexes)
    unresolved = [
        (case.id, strategy)
        for strategy in indexes
        for case in cases
        if case.answerable and not resolved.get((strategy, case.id))
    ]
    print(f"unresolved evidence: {unresolved if unresolved else 'none'}")

    # Retrieval sanity: how often is a gold chunk inside the dense top-20?
    for strategy in ("fixed", "structure"):
        hits = 0
        total = 0
        for case in cases:
            if not case.answerable:
                continue
            gold = resolved.get((strategy, case.id), set())
            if not gold:
                continue
            docs = dense_search(strategy, case.question, 20)
            ids = {str((d.metadata or {}).get("chunk_id")) for d in docs}
            total += 1
            if gold & ids:
                hits += 1
        print(f"dense top-20 recall on gold ({strategy}): {hits}/{total}")

    if collection_count("fixed") == 0 or collection_count("structure") == 0:
        print("FAIL: indexes are empty; run scripts/build_all_indexes.py --reset")
        raise SystemExit(1)
    print("preflight OK")


if __name__ == "__main__":
    main()
