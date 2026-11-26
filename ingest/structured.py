"""Structure-aware chunking index (Arm C and its prompt variants, C-1..C-4, C-q)."""

from __future__ import annotations

import argparse

from core.chunking import Chunk, structure_chunks
from core.config import settings
from core.parsing import parse_docs
from core.store import build_collection

from .common import attach_doc_metadata, doc_metadata_for, write_report
from core.console import enable_utf8


def build(reset: bool = False) -> dict:
    print(f"[ingest:structured] parsing {settings.raw_dir}")
    parsed = parse_docs(settings.raw_dir, cache_dir=settings.parsed_dir)
    chunks: list[Chunk] = []
    pages = 0
    for name, elements in sorted(parsed.items()):
        pages += len({e.page for e in elements})
        made = structure_chunks(
            elements,
            doc=name,
            doc_title=doc_metadata_for(name)["doc_title"],
            max_chars=settings.structure_chunk_size,
        )
        chunks.extend(made)
        print(f"  {name[:60]:60s} elements={len(elements):4d} chunks={len(made):3d}")
    attach_doc_metadata(chunks)
    print(f"[ingest:structured] {len(chunks)} chunks")
    build_collection("structure", chunks, reset=reset)
    return write_report(
        "structure",
        chunks,
        {"max_chars": settings.structure_chunk_size, "page_count": pages},
    )


def main() -> None:
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    print(build(reset=args.reset))


if __name__ == "__main__":
    main()
