"""Windows consoles default to GBK; printing German titles or Chinese paths
crashes with UnicodeEncodeError. Every CLI calls this once at start-up."""

from __future__ import annotations

import sys


def enable_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
