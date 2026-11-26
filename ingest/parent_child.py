"""Parent-child index (Arm C-c): 1800-char parents, 250-char sentence children."""

from __future__ import annotations

import argparse

from core.chunking import Chunk, parent_child_chunks
from core.config import settings
from core.parsing import parse_docs
from core.store import build_collection, parents_path, save_parents

from .common import attach_doc_metadata, doc_metadata_for, write_report
from core.console import enable_utf8


def build(reset: bool = False) -> dict:
    print(f"[ingest:parent_child] parsing {settings.raw_dir}")
    parsed = parse_docs(settings.raw_dir, cache_dir=settings.parsed_dir)
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for name, elements in sorted(parsed.items()):
        made_parents, made_children = parent_child_chunks(
            elements,
            doc=name,
            doc_title=doc_metadata_for(name)["doc_title"],
            parent_size=settings.parent_size,
            child_size=settings.child_size,
        )
        parents.extend(made_parents)
        children.extend(made_children)
        print(f"  {name[:60]:60s} parents={len(made_parents):3d} children={len(made_children):4d}")

    attach_doc_metadata(parents)
    attach_doc_metadata(children)
    path = save_parents(parents, parents_path("parent_child"))
    print(f"[ingest:parent_child] parents -> {path}")
    build_collection("parent_child", children, reset=reset)
    return write_report(
        "parent_child",
        children,
        {
            "parent_count": len(parents),
            "parent_size": settings.parent_size,
            "child_size": settings.child_size,
            "parents_path": str(path),
        },
    )


def main() -> None:
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    print(build(reset=args.reset))


if __name__ == "__main__":
    main()
