#!/usr/bin/env python3
"""
将公开数据集 **Labeled Faces in the Wild (LFW)** 中的人脸照写入各用户的 img0–img3。

依赖：scikit-learn（已存在于 requirements.txt）。首次运行会从网上下载 LFW（约百余 MB，需联网）。

用法（在项目根目录 visual-social-demo 下）::

    python scripts/populate_lfw_faces.py

可用选项::

    python scripts/populate_lfw_faces.py --overwrite

详见 data/DATASETS.md（引用格式与合规说明）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill data/images/<uid>/img*.png from LFW faces.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PNGs (default: skip if file exists).",
    )
    parser.add_argument(
        "--min-faces",
        type=int,
        default=4,
        help="Minimum photos per identity in LFW subset (default 4).",
    )
    parser.add_argument(
        "--resize-quality",
        type=float,
        default=0.85,
        help="sklearn LFW resize ratio for download pipeline (default 0.85).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from PIL import Image

    from src.io_profiles import ensure_profiles_file

    ensure_profiles_file(root)
    users_path = root / "data" / "users.json"
    payload = json.loads(users_path.read_text(encoding="utf-8"))
    users = payload["users"]

    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError as e:
        raise SystemExit("请安装 scikit-learn：pip install scikit-learn") from e

    print("正在下载/加载 LFW（首次较慢）…")
    lfw = fetch_lfw_people(
        min_faces_per_person=max(4, args.min_faces),
        color=True,
        resize=args.resize_quality,
        download_if_missing=True,
    )

    # 按身份聚合索引；每人至少 min_faces 张（已由 fetch 过滤）
    from collections import defaultdict

    by_person: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(lfw.target):
        by_person[int(label)].append(int(idx))

    eligible = sorted(pid for pid in by_person if len(by_person[pid]) >= 4)
    if len(eligible) < len(users):
        raise SystemExit(
            f"LFW 子集中仅有 {len(eligible)} 个身份满足每人≥4张图，少于用户数 {len(users)}。"
            "可降低 --min-faces（不小于 4）或改用其它数据集。"
        )

    selected = eligible[: len(users)]

    manifest: list[dict[str, object]] = []
    target_names = list(lfw.target_names)

    for slot, u in enumerate(users):
        pid = selected[slot]
        indices = by_person[pid][:4]
        uid = u["user_id"]
        name = str(target_names[pid]) if pid < len(target_names) else str(pid)
        manifest.append(
            {
                "user_id": uid,
                "lfw_person_label_index": pid,
                "lfw_person_name": name,
                "lfw_image_indices": indices,
            }
        )

        for j, img_idx in enumerate(indices):
            rel = u["images"][j]
            out_path = root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.overwrite:
                continue

            arr = lfw.images[img_idx]
            if arr.ndim == 2:
                rgb = arr
                im = Image.fromarray((rgb * 255).clip(0, 255).astype("uint8"), mode="L").convert("RGB")
            else:
                rgb = arr
                im = Image.fromarray((rgb * 255).clip(0, 255).astype("uint8"), mode="RGB")
            im = im.resize((384, 384), Image.Resampling.LANCZOS)
            im.save(out_path, format="PNG")

    manifest_path = root / "data" / "lfw_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成：已写入人脸图（详见映射 {manifest_path}）。同一用户的 4 张图来自同一 LFW 身份。")
    print("下一步（推荐）：python scripts/merge_lfw_stated_profiles.py --backup")
    print("  → 昵称用 LFW，兴趣/偏好恢复为题目要求的模拟自述；视觉画像仍由程序对图实时生成。")


if __name__ == "__main__":
    main()
