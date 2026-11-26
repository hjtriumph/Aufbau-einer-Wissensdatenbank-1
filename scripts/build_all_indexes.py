"""Build the three indexes in the order the arms expect them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest import basic, parent_child, structured
from core.console import enable_utf8


def main() -> None:
    enable_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--only", choices=["fixed", "structure", "parent_child"], default="")
    args = ap.parse_args()

    if args.only in ("", "fixed"):
        print(basic.build(reset=args.reset))
    if args.only in ("", "structure"):
        print(structured.build(reset=args.reset))
    if args.only in ("", "parent_child"):
        print(parent_child.build(reset=args.reset))


if __name__ == "__main__":
    main()
