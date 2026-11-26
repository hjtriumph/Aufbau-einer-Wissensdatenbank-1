"""Dump per-page plain text for benchmark authoring and quote validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import settings
from core.parsing import page_texts


def main() -> None:
    out_dir = settings.corpus_dir / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for path in sorted(settings.raw_dir.glob("*.pdf")):
        pages = page_texts(path)
        target = out_dir / f"{path.stem}.txt"
        blocks = [f"\n===== PAGE {i} =====\n{text}" for i, text in enumerate(pages, start=1)]
        target.write_text("".join(blocks), encoding="utf-8")
        index[path.name] = {"text": target.name, "pages": len(pages)}
        print(f"{path.name[:70]:70s} pages={len(pages)}")
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[extract] {len(index)} files -> {out_dir}")


if __name__ == "__main__":
    main()
