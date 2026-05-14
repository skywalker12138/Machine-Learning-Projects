"""Load / persist user profiles JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_profiles_file(root: Path | None = None) -> Path:
    """Write data/users.json from builtins if missing; return path."""
    root = root or project_root()
    path = root / "data" / "users.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        from .builtin_users import iter_builtin_profiles

        payload = {"users": list(iter_builtin_profiles())}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_profiles(root: Path | None = None) -> list[dict[str, Any]]:
    ensure_profiles_file(root)
    root = root or project_root()
    path = root / "data" / "users.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["users"])
