#!/usr/bin/env python3
"""CLI wrapper: generate placeholder PNGs + ensure users.json exists."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.bootstrap_images import ensure_placeholder_images

    ensure_placeholder_images(root)
    print(f"OK: ensured images under {root / 'data' / 'images'}")


if __name__ == "__main__":
    main()
