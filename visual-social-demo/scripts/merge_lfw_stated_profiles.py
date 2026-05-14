#!/usr/bin/env python3
"""
将「题目要求的用户自述」与 LFW 显示名合并写回 users.json。

- **昵称**：来自 data/lfw_manifest.json（LFW 身份名，仅作展示，非视觉分析结果）。
- **兴趣标签、自然语言偏好、preference_mode**：恢复为内置 ROWS（与最初 Demo 人设一致，满足「多用户资料输入」）。
- **images**：不变。

运行顺序建议::

    python scripts/populate_lfw_faces.py --overwrite
    python scripts/merge_lfw_stated_profiles.py --backup

不再使用「把 preference 写成 LFW 说明」的旧逻辑；视觉画像一律由程序对图片实时计算（见 app 界面「系统视觉画像」区块）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.builtin_users import iter_builtin_profiles

    manifest_path = root / "data" / "lfw_manifest.json"
    users_path = root / "data" / "users.json"
    if not manifest_path.exists():
        raise SystemExit(f"缺少 {manifest_path}，请先运行 populate_lfw_faces.py")

    stated = {u["user_id"]: u for u in iter_builtin_profiles()}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_uid = {m["user_id"]: m for m in manifest}

    payload = json.loads(users_path.read_text(encoding="utf-8"))
    users = payload["users"]

    if args.backup:
        shutil.copy2(users_path, users_path.with_suffix(".json.bak"))

    for u in users:
        uid = u["user_id"]
        if uid not in stated:
            raise SystemExit(f"内置人设缺少 {uid}")
        if uid not in by_uid:
            raise SystemExit(f"lfw_manifest 缺少 {uid}")
        base = stated[uid]
        u["nickname"] = str(by_uid[uid].get("lfw_person_name", base["nickname"]))[:80]
        u["interest_tags"] = list(base["interest_tags"])
        u["preference_text"] = str(base["preference_text"])
        u["preference_mode"] = str(base["preference_mode"])

    users_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已合并：LFW 显示名 + 内置兴趣/偏好/模式；视觉画像仍由运行时图像分析生成。")


if __name__ == "__main__":
    main()
