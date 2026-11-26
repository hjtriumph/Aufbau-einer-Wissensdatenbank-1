from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .parsing import Element
from .textnorm import normalize_text

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+(?=[A-ZÄÖÜ0-9\"“(])")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return str(self.metadata["chunk_id"])


def stable_chunk_id(doc: str, page_start: int, text: str, ordinal: int = 0) -> str:
    """Content-addressed id with a positional disambiguator.

    Identical text can legitimately appear twice on one page (repeated table
    headers, boilerplate); without the ordinal the second copy would collide and
    Chroma's upsert would silently drop it.
    """
    raw = f"{doc}|{page_start}|{ordinal}|{normalize_text(text)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _base_metadata(doc: str, doc_title: str) -> dict:
    return {"doc": doc, "doc_title": doc_title}


def fixed_page_chunks(
    page_numbers: Sequence[int],
    page_texts: Sequence[str],
    doc: str,
    doc_title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Fixed recursive chunking, page by page (Arm B / D / D' / C-2..C-4 index).

    Page-local splitting keeps a page number on every chunk, so gold evidence
    can be resolved by (page, quote) without cross-page ambiguity.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out: list[Chunk] = []
    for page_no, text in zip(page_numbers, page_texts):
        for ordinal, piece in enumerate(splitter.split_text(text or "")):
            piece = piece.strip()
            if len(piece) < 40:
                continue
            cid = stable_chunk_id(doc, page_no, piece, ordinal)
            out.append(
                Chunk(
                    text=piece,
                    metadata={
                        **_base_metadata(doc, doc_title),
                        "chunk_id": cid,
                        "page_start": page_no,
                        "page_end": page_no,
                        "chunk_type": "text",
                        "heading_path": "",
                        "strategy": "fixed",
                    },
                )
            )
    for index, chunk in enumerate(out):
        chunk.metadata["chunk_index"] = index
    return out


def _heading_path(stack: Sequence[tuple[int, str]]) -> str:
    return " > ".join(text for _, text in stack)


def _merge_undersized(chunks: list[Chunk], min_chars: int, max_chars: int) -> list[Chunk]:
    """Merge a short chunk into the previous one when they share a heading path."""
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged:
            prev = merged[-1]
            same_section = prev.metadata.get("heading_path") == chunk.metadata.get("heading_path")
            if (
                same_section
                and len(prev.text) < min_chars
                and len(prev.text) + len(chunk.text) + 2 <= max_chars
            ):
                prev.text = f"{prev.text}\n\n{chunk.text}"
                prev.metadata["page_end"] = chunk.metadata.get("page_end", prev.metadata["page_end"])
                continue
        merged.append(chunk)
    return merged


def structure_chunks(
    elements: Sequence[Element],
    doc: str,
    doc_title: str,
    max_chars: int,
    min_chars: int = 200,
) -> list[Chunk]:
    """Structure-aware chunking: sections start at headings, tables stay atomic."""
    stack: list[tuple[int, str]] = []
    body: list[Element] = []
    body_chars = 0
    out: list[Chunk] = []

    def emit(kind: str = "text") -> None:
        nonlocal body, body_chars
        if not body:
            return
        path = _heading_path(stack)
        text = "\n\n".join(e.text for e in body)
        full = f"{path}\n\n{text}" if path else text
        page_start = min(e.page for e in body)
        page_end = max(e.page for e in body)
        out.append(
            Chunk(
                text=full,
                metadata={
                    **_base_metadata(doc, doc_title),
                    "chunk_id": stable_chunk_id(doc, page_start, full, len(out)),
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_type": kind,
                    "heading_path": path,
                    "strategy": "structure",
                },
            )
        )
        body = []
        body_chars = 0

    for element in elements:
        if element.type == "heading" and element.level:
            emit()
            while stack and stack[-1][0] >= element.level:
                stack.pop()
            stack.append((element.level, element.text.strip()))
            continue

        if element.type == "table":
            emit()
            path = _heading_path(stack)
            text = f"{path}\n\n{element.text}" if path else element.text
            out.append(
                Chunk(
                    text=text,
                    metadata={
                        **_base_metadata(doc, doc_title),
                        "chunk_id": stable_chunk_id(doc, element.page, text, len(out)),
                        "page_start": element.page,
                        "page_end": element.page,
                        "chunk_type": "table",
                        "heading_path": path,
                        "strategy": "structure",
                    },
                )
            )
            continue

        # captions ride with the following paragraph they describe
        if body and body_chars + len(element.text) + 2 > max_chars:
            emit()
        body.append(element)
        body_chars += len(element.text) + 2

    emit()

    out = _merge_undersized(out, min_chars=min_chars, max_chars=max_chars)
    for index, chunk in enumerate(out):
        chunk.metadata["chunk_index"] = index
    return out


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split((text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _windows(text: str, size: int) -> list[str]:
    sentences = _split_sentences(text)
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            windows.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
        # an over-long sentence is hard-split so no child exceeds ~1.3x the size
        while len(current) > int(size * 1.3):
            windows.append(current[:size].strip())
            current = current[size:].strip()
    if current.strip():
        windows.append(current.strip())
    return windows


def parent_child_chunks(
    elements: Sequence[Element],
    doc: str,
    doc_title: str,
    parent_size: int,
    child_size: int,
) -> tuple[list[Chunk], list[Chunk]]:
    """Parent sections (≤parent_size) + sentence-window children (≤child_size)."""
    parents = structure_chunks(
        elements, doc=doc, doc_title=doc_title, max_chars=parent_size, min_chars=0
    )

    children: list[Chunk] = []
    for parent in parents:
        parent_text = parent.text
        path = str(parent.metadata.get("heading_path", ""))
        body = parent_text[len(path) :].strip() if path and parent_text.startswith(path) else parent_text
        for window_index, window in enumerate(_windows(body, child_size)):
            text = f"{path}\n\n{window}" if path else window
            cid = hashlib.sha1(
                f"{parent.chunk_id}|{window_index}|{normalize_text(window)}".encode("utf-8")
            ).hexdigest()[:16]
            children.append(
                Chunk(
                    text=text,
                    metadata={
                        **_base_metadata(doc, doc_title),
                        "chunk_id": cid,
                        "parent_id": parent.chunk_id,
                        "page_start": parent.metadata.get("page_start"),
                        "page_end": parent.metadata.get("page_end"),
                        "chunk_type": "child",
                        "heading_path": path,
                        "strategy": "parent_child",
                    },
                )
            )

    for index, chunk in enumerate(children):
        chunk.metadata["chunk_index"] = index
    for index, parent in enumerate(parents):
        parent.metadata["chunk_index"] = index
    return parents, children


def fixed_chunks_dir(
    raw_dir: str | Path,
    doc_title_for,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Build the fixed index straight from PDF page text (PyPDFLoader-equivalent)."""
    from pypdf import PdfReader

    out: list[Chunk] = []
    for path in sorted(Path(raw_dir).glob("*.pdf")):
        reader = PdfReader(str(path))
        texts = [(page.extract_text() or "") for page in reader.pages]
        pages = list(range(1, len(texts) + 1))
        out.extend(
            fixed_page_chunks(
                pages,
                texts,
                doc=path.name,
                doc_title=doc_title_for(path.name),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    for index, chunk in enumerate(out):
        chunk.metadata["chunk_index"] = index
    return out
