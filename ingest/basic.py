"""Fixed 1000/150 recursive chunking index (Arms B, D, D', C-2, C-3, C-4)."""

from __future__ import annotations

import argparse

from core.chunking import fixed_chunks_dir
from core.config import settings
from core.store import build_collection

from .common import attach_doc_metadata, doc_title_for, write_report
from core.console import enable_utf8


def build(reset: bool = False) -> dict:
    print(f"[ingest:basic] {settings.raw_dir} -> fixed {settings.fixed_chunk_size}/{settings.fixed_chunk_overlap}")
    chunks = fixed_chunks_dir(
        settings.raw_dir,
        doc_title_for=doc_title_for,
        chunk_size=settings.fixed_chunk_size,
        chunk_overlap=settings.fixed_chunk_overlap,
    )
    attach_doc_metadata(chunks)
    print(f"[ingest:basic] {len(chunks)} chunks, embedding with {settings.embedding_model}")
    build_collection("fixed", chunks, reset=reset)
    return write_report(
        "fixed",
        chunks,
        {
            "chunk_size": settings.fixed_chunk_size,
            "chunk_overlap": settings.fixed_chunk_overlap,
        },
    )


def main() -> None:
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    report = build(reset=args.reset)
    print(report)


if __name__ == "__main__":
    main()
