"""Offline check: can every gold quote be resolved to a chunk in each strategy?

Uses the chunkers directly (no embeddings, no Chroma), so it runs in seconds
after the parse cache exists.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.chunking import Chunk, fixed_chunks_dir, parent_child_chunks, structure_chunks
from core.config import settings
from core.parsing import parse_docs
from core.console import enable_utf8
from ingest.common import doc_metadata_for, doc_title_for
from eval.case_loader import load_cases, resolve_evidence


def build_indexes() -> dict[str, list[dict]]:
    fixed = fixed_chunks_dir(
        settings.raw_dir, doc_title_for, settings.fixed_chunk_size, settings.fixed_chunk_overlap
    )
    parsed = parse_docs(settings.raw_dir, cache_dir=settings.parsed_dir)
    structured: list[Chunk] = []
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for name, elements in sorted(parsed.items()):
        title = doc_metadata_for(name)["doc_title"]
        structured.extend(
            structure_chunks(elements, name, title, max_chars=settings.structure_chunk_size)
        )
        p, c = parent_child_chunks(
            elements, name, title, parent_size=settings.parent_size, child_size=settings.child_size
        )
        parents.extend(p)
        children.extend(c)

    def to_index(chunks, parent_level=False):
        out = []
        for chunk in chunks:
            meta = dict(chunk.metadata)
            if parent_level:
                meta["parent_id"] = meta["chunk_id"]
            out.append({"chunk_id": meta["chunk_id"], "text": chunk.text, "metadata": meta})
        return out

    return {
        "fixed": to_index(fixed),
        "structure": to_index(structured),
        "parent_child": to_index(children),
        "parents": to_index(parents, parent_level=True),
    }


def main() -> None:
    enable_utf8()
    indexes = build_indexes()
    for name, index in indexes.items():
        print(f"index {name}: {len(index)}")

    cases = load_cases(settings.questions_path)
    for strategy, index_name, parent_level in (
        ("fixed", "fixed", False),
        ("structure", "structure", False),
        ("parent_child", "parents", True),
    ):
        methods: Counter[str] = Counter()
        missing = []
        for case in cases:
            if not case.answerable:
                continue
            ids, method = resolve_evidence(
                case, indexes[index_name], ignore_page=parent_level, parent_level=parent_level
            )
            methods[method] += 1
            if not ids:
                missing.append(case.id)
        print(f"{strategy}: {dict(methods)} unresolved={missing}")


if __name__ == "__main__":
    main()
