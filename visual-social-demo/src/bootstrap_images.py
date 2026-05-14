"""Create deterministic placeholder PNGs referenced by profiles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .io_profiles import ensure_profiles_file


def _color_from_seed(seed: str) -> tuple[int, int, int]:
    h = hashlib.sha256(seed.encode()).digest()
    return int(h[0]), int(h[1]), int(h[2])


def ensure_placeholder_images(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[1]
    ensure_profiles_file(root)
    profiles = json.loads((root / "data" / "users.json").read_text(encoding="utf-8"))["users"]

    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise RuntimeError("Install Pillow to generate placeholder images.") from e

    for u in profiles:
        uid = u["user_id"]
        for i, rel in enumerate(u["images"]):
            p = root / rel
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            idx = Path(rel).stem
            seed = f"{uid}:{idx}"
            base = _color_from_seed(seed)
            img = Image.new("RGB", (384, 384), base)
            draw = ImageDraw.Draw(img)
            accent = tuple((c + 80 + i * 17) % 255 for c in base)
            draw.rectangle([96, 96, 288, 288], outline=accent, width=8)
            draw.ellipse([140, 140, 244, 244], fill=accent)
            img.save(p, format="PNG")
