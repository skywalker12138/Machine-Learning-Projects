"""Aggregate multi-image features into a single visual persona per user."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.io_profiles import project_root
from src.vision.extractor import (
    FrameFeatures,
    VisionExtractor,
    aggregate_ocr_snippets,
    aggregate_prompt_scores,
    aggregate_yolo_tags,
    top_prompt_tags,
)
from src.vision.prompt_taxonomy import top_in_buckets


@dataclass
class UserPersona:
    """用户画像 = 自述字段（来自 users.json）+ 系统视觉画像（本结构体中除自述外的字段均为图像流水线实时生成）。"""

    user_id: str
    nickname: str
    interest_tags: list[str]
    preference_text: str
    preference_mode: str
    clip_centroid: np.ndarray
    prompt_scores: dict[str, float]
    top_visual_prompts: list[tuple[str, float]]
    # --- 以下由 VisionExtractor 对多图聚合得到（满足题目：场景/物体/穿搭/活动 + OCR 线索）---
    scene_clues: list[tuple[str, float]]
    activity_clues: list[tuple[str, float]]
    outfit_clues: list[tuple[str, float]]
    object_clues: list[tuple[str, float]]  # YOLO 物体检测
    ocr_clues: list[str]
    yolo_tags: list[tuple[str, float]]
    preference_embedding: np.ndarray


def _resolve_paths(root: Path, rels: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for r in rels:
        p = (root / r).resolve()
        out.append(p)
    return out


def build_persona_for_user(
    user: dict[str, Any],
    extractor: VisionExtractor,
    root: Path | None = None,
) -> UserPersona:
    root = root or project_root()
    paths = _resolve_paths(root, user["images"])
    frames: list[FrameFeatures] = []
    vecs: list[np.ndarray] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing image for {user['user_id']}: {p}")
        ff = extractor.extract_frame(p)
        frames.append(ff)
        vecs.append(ff.clip_vec)

    centroid = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    pscores = aggregate_prompt_scores(frames)
    top_vp = top_prompt_tags(pscores, k=12)
    ytags = aggregate_yolo_tags(frames, max_tags=12)
    scene_top = top_in_buckets(pscores, "scene", k=8)
    activity_top = top_in_buckets(pscores, "activity", k=8)
    outfit_top = top_in_buckets(pscores, "outfit", k=8)
    ocr_lines = aggregate_ocr_snippets(frames, max_lines=8)

    pref_emb = extractor.embed_text_single(_preference_augmented_text(user))

    return UserPersona(
        user_id=str(user["user_id"]),
        nickname=str(user["nickname"]),
        interest_tags=list(user["interest_tags"]),
        preference_text=str(user["preference_text"]),
        preference_mode=str(user["preference_mode"]),
        clip_centroid=centroid,
        prompt_scores=pscores,
        top_visual_prompts=top_vp,
        scene_clues=scene_top,
        activity_clues=activity_top,
        outfit_clues=outfit_top,
        object_clues=list(ytags),
        ocr_clues=ocr_lines,
        yolo_tags=ytags,
        preference_embedding=pref_emb.astype(np.float32),
    )


def _preference_augmented_text(user: dict[str, Any]) -> str:
    tags = " ".join(user.get("interest_tags", []))
    pref = user.get("preference_text", "")
    mode = user.get("preference_mode", "similar")
    return f"{pref}\n兴趣标签：{tags}\n匹配模式：{mode}"


def build_all_personas(users: list[dict[str, Any]], extractor: VisionExtractor, root: Path | None = None) -> dict[str, UserPersona]:
    root = root or project_root()
    personas: dict[str, UserPersona] = {}
    for u in users:
        personas[str(u["user_id"])] = build_persona_for_user(u, extractor, root=root)
    return personas


def persona_to_tag_weights(persona: UserPersona) -> dict[str, float]:
    """Flatten visual tags for overlap / explanations."""
    w: dict[str, float] = {}
    for t, s in persona.top_visual_prompts:
        w[f"视觉:{t}"] = float(s)
    for t, c in persona.yolo_tags:
        w[t] = float(c / (sum(x for _, x in persona.yolo_tags) + 1e-9))
    for t in persona.interest_tags:
        w[f"兴趣:{t}"] = w.get(f"兴趣:{t}", 0.0) + 1.0
    return w
