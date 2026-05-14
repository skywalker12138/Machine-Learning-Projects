#!/usr/bin/env python3
"""Rewrite data/users.json from builtins (optional refresh)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.builtin_users import iter_builtin_profiles

    out = root / "data" / "users.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"users": list(iter_builtin_profiles())}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload['users'])} users to {out}")


if __name__ == "__main__":
    main()
