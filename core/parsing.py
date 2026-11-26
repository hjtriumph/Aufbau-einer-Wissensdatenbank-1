from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

_HEADING_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)[\.\s]")
_ANY_NUMBER_RE = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*|[IVXLC]+\.|[A-Z]\.)\s")
_CAPTION_RE = re.compile(r"^(fig(?:ure)?\.?|table|abbildung|tabelle)\s*\d+", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"^[ivxlcdm]{1,7}$|^\d{1,4}$", re.IGNORECASE)
_FORMULA_RE = re.compile(r"[=<>≤≥∑∫√]")
_SENTENCE_END = ".,;，。；:"
_PARSER_VERSION = 3

_BOILERPLATE_RES = (
    re.compile(r"^contents lists available at", re.IGNORECASE),
    re.compile(r"journal homepage:", re.IGNORECASE),
    re.compile(r"^date of publication", re.IGNORECASE),
    re.compile(r"digital object identifier", re.IGNORECASE),
    re.compile(r"^(?:©|\(c\))\s*\d{4}", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"^issn\b", re.IGNORECASE),
    re.compile(r"^https?://doi\.org", re.IGNORECASE),
    re.compile(r"^available online", re.IGNORECASE),
    re.compile(r"^this is an open access article", re.IGNORECASE),
    re.compile(r"^received\b.{0,80}\baccepted\b", re.IGNORECASE),
)


def is_boilerplate(text: str) -> bool:
    t = " ".join((text or "").split())
    if not t:
        return False
    return any(pattern.search(t) for pattern in _BOILERPLATE_RES)


@dataclass
class Element:
    type: str  # heading | paragraph | caption | table
    text: str
    page: int  # 1-based, user-facing
    bbox: tuple[float, float, float, float] | None = None
    level: int | None = None


def detect_heading(text: str, size: float, body_size: float, bold: bool) -> int | None:
    t = " ".join((text or "").split())
    if not t or len(t) > 140:
        return None
    if t[-1] in _SENTENCE_END:
        return None
    if t[0].islower():
        return None
    if _FORMULA_RE.search(t):
        return None
    if len(t) <= 3 and t.isupper():
        return None

    numbered = _HEADING_NUMBER_RE.match(t)
    emphasized = bold or (body_size > 0 and size >= body_size * 1.05)
    if numbered:
        if not emphasized:
            return None
        return min(4, numbered.group(1).count(".") + 1)

    ratio = size / body_size if body_size > 0 else 1.0
    if ratio >= 1.4:
        return 1
    if ratio >= 1.2:
        return 2
    if len(t) <= 90 and (bold or t.isupper()) and t[0].isalnum():
        return 2
    return None


def join_lines(lines: Sequence[str]) -> str:
    out = ""
    for line in lines:
        piece = (line or "").strip()
        if not piece:
            continue
        if not out:
            out = piece
        elif out.endswith("-") and piece[:1].islower():
            out = out[:-1] + piece
        else:
            out = f"{out} {piece}"
    return out


def table_to_markdown(rows: Sequence[Sequence[object]]) -> str:
    cleaned: list[list[str]] = []
    for row in rows or []:
        cleaned.append(
            [("" if cell is None else str(cell)).strip().replace("|", "\\|") for cell in row]
        )
    if not cleaned:
        return ""
    n_cols = max(len(row) for row in cleaned)
    for row in cleaned:
        row.extend([""] * (n_cols - len(row)))
    lines = ["| " + " | ".join(row) + " |" for row in [cleaned[0], ["---"] * n_cols, *cleaned[1:]]]
    return "\n".join(lines)


def merge_wrapped_headings(elements: Sequence[Element]) -> list[Element]:
    merged: list[Element] = []
    for element in elements:
        if merged:
            previous = merged[-1]
            if (
                element.type == "heading"
                and previous.type == "heading"
                and element.level == previous.level
                and element.page == previous.page
                and not _ANY_NUMBER_RE.match(previous.text)
                and not _ANY_NUMBER_RE.match(element.text)
                and previous.text[-1] not in _SENTENCE_END
                and element.text[:1].isupper()
                and len(previous.text) < 100
            ):
                merged[-1] = replace(previous, text=f"{previous.text} {element.text}")
                continue
        merged.append(element)
    return merged


def _two_column(bboxes: Sequence[tuple[float, float, float, float]], page_width: float) -> bool:
    left = sum(1 for b in bboxes if b[2] <= page_width * 0.58)
    right = sum(1 for b in bboxes if b[0] >= page_width * 0.42)
    return left >= 2 and right >= 2


def _order_key(bbox: tuple[float, float, float, float], page_width: float, two_col: bool) -> tuple:
    x0, y0, x1, _ = bbox
    if not two_col:
        return (round(y0 / 8), x0)
    center = (x0 + x1) / 2
    return (round(y0 / 8), 0 if center < page_width / 2 else 1, x0)


def order_elements(elements: Sequence[Element], page_width: float) -> list[Element]:
    if not elements:
        return []
    bboxes = [e.bbox or (0.0, 0.0, 0.0, 0.0) for e in elements]
    two_col = _two_column(bboxes, page_width)
    return sorted(
        elements,
        key=lambda e: _order_key(e.bbox or (0.0, 0.0, 0.0, 0.0), page_width, two_col),
    )


def _dominant_size(spans: Sequence[dict]) -> float:
    weighted: list[float] = []
    for span in spans:
        n = max(1, len(span.get("text", "")))
        weighted.extend([float(span.get("size", 0) or 0)] * min(n, 50))
    return statistics.median(weighted) if weighted else 0.0


def _bold_ratio(spans: Sequence[dict]) -> float:
    total = sum(len(s.get("text", "")) for s in spans) or 1
    bold = sum(len(s.get("text", "")) for s in spans if int(s.get("flags", 0)) & 16)
    return bold / total


def _body_size(lines: Sequence[dict]) -> float:
    sizes: list[float] = []
    for line in lines:
        size = float(line.get("size", 0) or 0)
        if size > 0:
            sizes.extend([size] * max(1, min(len(line.get("text", "")), 50)))
    return statistics.median(sizes) if sizes else 0.0


def _keep_mask(pages_text: Sequence[Sequence[str]], min_ratio: float = 0.5) -> list[list[bool]]:
    counts = Counter(t for page in pages_text for t in page if t)
    threshold = max(2, math.ceil(min_ratio * max(1, len(pages_text))))
    return [
        [bool(t) and not _PAGE_NUMBER_RE.match(t) and counts[t] < threshold for t in page]
        for page in pages_text
    ]


def _inside_any_table(
    bbox: tuple[float, float, float, float], tables: Sequence[dict]
) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for table in tables:
        tx0, ty0, tx1, ty1 = table["bbox"]
        if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
            return True
    return False


def _lines_to_elements(lines: Sequence[dict], page_no: int, body_size: float) -> list[Element]:
    para: list[str] = []
    elements: list[Element] = []
    prev_bottom: float | None = None
    para_bbox: tuple[float, float, float, float] | None = None

    def flush_paragraph() -> None:
        nonlocal para, para_bbox
        text = join_lines(para).strip()
        if text and not is_boilerplate(text):
            elements.append(
                Element("paragraph", text, page_no, bbox=para_bbox if para_bbox else None)
            )
        para = []
        para_bbox = None

    for line in lines:
        text = line.get("text", "")
        bbox = line.get("bbox", (0, 0, 0, 0))
        if is_boilerplate(text):
            continue
        level = detect_heading(
            text,
            size=float(line.get("size", 0) or 0),
            body_size=body_size,
            bold=bool(line.get("bold")),
        )
        if level is not None:
            flush_paragraph()
            elements.append(Element("heading", text.strip(), page_no, bbox=bbox, level=level))
            prev_bottom = bbox[3]
            continue

        if _CAPTION_RE.match(text.strip()) and len(text) < 300:
            flush_paragraph()
            elements.append(Element("caption", text.strip(), page_no, bbox=bbox))
            prev_bottom = bbox[3]
            continue

        gap = float("inf") if prev_bottom is None else bbox[1] - prev_bottom
        line_height = max(1.0, bbox[3] - bbox[1])
        if para and gap > 1.8 * line_height:
            flush_paragraph()
        if not para:
            para_bbox = bbox
        para.append(text)
        prev_bottom = bbox[3]

    flush_paragraph()
    return elements


def parse_pdf(path: str | Path, detect_tables: bool = True) -> list[Element]:
    import pymupdf

    doc = pymupdf.open(str(path))
    pages: list[dict] = []
    try:
        for page_index, page in enumerate(doc, start=1):
            data = page.get_text("dict")
            lines: list[dict] = []
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans") or []
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    lines.append(
                        {
                            "text": text,
                            "size": _dominant_size(spans),
                            "bold": _bold_ratio(spans) >= 0.6,
                            "bbox": tuple(line.get("bbox", (0.0, 0.0, 0.0, 0.0))),
                        }
                    )

            tables: list[dict] = []
            if detect_tables:
                try:
                    for table in page.find_tables().tables:
                        rows = table.extract()
                        if rows:
                            tables.append(
                                {"bbox": tuple(table.bbox), "markdown": table_to_markdown(rows)}
                            )
                except Exception:
                    tables = []

            pages.append(
                {
                    "page": page_index,
                    "lines": lines,
                    "tables": tables,
                    "width": float(page.rect.width),
                }
            )
    finally:
        doc.close()

    if pages:
        mask = _keep_mask([[l["text"] for l in p["lines"]] for p in pages])
        for page, keep in zip(pages, mask):
            page["lines"] = [l for l, ok in zip(page["lines"], keep) if ok]

    elements: list[Element] = []
    for page in pages:
        body_size = _body_size(page["lines"]) or 10.0
        page_elements = _lines_to_elements(page["lines"], page["page"], body_size)

        if page["tables"]:
            page_elements = [
                e for e in page_elements if not (e.bbox and _inside_any_table(e.bbox, page["tables"]))
            ]
            for table in page["tables"]:
                page_elements.append(
                    Element("table", table["markdown"], page["page"], bbox=table["bbox"])
                )

        elements.extend(merge_wrapped_headings(order_elements(page_elements, page["width"])))

    return elements


def _to_records(elements: Sequence[Element]) -> list[dict]:
    return [
        {
            "type": e.type,
            "text": e.text,
            "page": e.page,
            "bbox": list(e.bbox) if e.bbox else None,
            "level": e.level,
        }
        for e in elements
    ]


def _from_records(records: Sequence[dict]) -> list[Element]:
    return [
        Element(
            r["type"],
            r["text"],
            int(r["page"]),
            bbox=tuple(r["bbox"]) if r.get("bbox") else None,
            level=r.get("level"),
        )
        for r in records
    ]


def parse_docs(
    raw_dir: str | Path,
    cache_dir: str | Path | None = None,
    detect_tables: bool = True,
) -> dict[str, list[Element]]:
    """Parse every PDF under raw_dir, keyed by basename. Cached by mtime/size."""
    out: dict[str, list[Element]] = {}
    cache_root = Path(cache_dir) if cache_dir else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)

    for path in sorted(Path(raw_dir).glob("*.pdf")):
        stat = path.stat()
        meta = {
            "version": _PARSER_VERSION,
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "detect_tables": bool(detect_tables),
        }
        cache_file = cache_root / f"{path.stem}.json" if cache_root else None

        if cache_file and cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                if payload.get("meta") == meta:
                    out[path.name] = _from_records(payload["elements"])
                    continue
            except Exception:
                pass

        elements = parse_pdf(path, detect_tables=detect_tables)
        out[path.name] = elements
        if cache_file:
            cache_file.write_text(
                json.dumps({"meta": meta, "elements": _to_records(elements)}, ensure_ascii=False),
                encoding="utf-8",
            )

    return out


def page_texts(path: str | Path) -> list[str]:
    """Plain per-page text, used for benchmark authoring and quote validation."""
    import pymupdf

    doc = pymupdf.open(str(path))
    try:
        return [page.get_text("text") for page in doc]
    finally:
        doc.close()
