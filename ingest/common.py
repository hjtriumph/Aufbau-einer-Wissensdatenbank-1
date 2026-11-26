from __future__ import annotations

import json
from pathlib import Path

from core.chunking import Chunk
from core.config import settings
from core.store import collection_count


def documents_registry() -> dict[str, dict]:
    path = settings.corpus_dir / "documents.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def doc_title_for(filename: str) -> str:
    entry = documents_registry().get(filename) or {}
    return str(entry.get("title") or Path(filename).stem)


def doc_metadata_for(filename: str) -> dict:
    entry = documents_registry().get(filename) or {}
    return {
        "doc_title": str(entry.get("title") or Path(filename).stem),
        "doc_language": str(entry.get("language") or ""),
        "doc_authors": str(entry.get("authors") or ""),
        "doc_year": str(entry.get("year") or ""),
        "doc_venue": str(entry.get("venue") or ""),
    }


def attach_doc_metadata(chunks: list[Chunk]) -> list[Chunk]:
    """Copy bibliographic fields onto every chunk (citation cards read them)."""
    registry = documents_registry()
    for chunk in chunks:
        doc = str(chunk.metadata.get("doc", ""))
        entry = registry.get(doc) or {}
        chunk.metadata.update(
            {
                "doc_language": str(entry.get("language") or ""),
                "doc_authors": str(entry.get("authors") or ""),
                "doc_year": str(entry.get("year") or ""),
                "doc_venue": str(entry.get("venue") or ""),
                "doc_doi": str(entry.get("doi") or ""),
            }
        )
    return chunks


def write_report(strategy: str, chunks: list[Chunk], extra: dict | None = None) -> dict:
    docs = sorted({str(c.metadata.get("doc", "")) for c in chunks})
    ids = [str(c.metadata["chunk_id"]) for c in chunks]
    report = {
        "strategy": strategy,
        "collection": strategy,
        "document_count": len(docs),
        "chunk_count": collection_count(strategy),
        "expected_chunk_count": len(chunks),
        "duplicate_ids": len(ids) - len(set(ids)),
        "documents": docs,
    }
    report.update(extra or {})
    path = settings.chroma_dir.parent / f"ingest_report_{strategy}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest:{strategy}] report -> {path}")
    return report
