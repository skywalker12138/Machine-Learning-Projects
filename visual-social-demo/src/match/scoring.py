"""Similar / complementary scoring between personas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.profile.builder import UserPersona


def _cos01(a: np.ndarray, b: np.ndarray) -> float:
    v = float(np.dot(a, b))
    return float(np.clip((v + 1.0) / 2.0, 0.0, 1.0))


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.5
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0


def _prompt_overlap_score(q: UserPersona, c: UserPersona, top_k: int = 8) -> float:
    qa = [t for t, _ in sorted(q.top_visual_prompts, key=lambda x: x[1], reverse=True)[:top_k]]
    ca = [t for t, _ in sorted(c.top_visual_prompts, key=lambda x: x[1], reverse=True)[:top_k]]
    sa, sb = set(qa), set(ca)
    if not sa and not sb:
        return 0.5
    return float(len(sa & sb) / len(sa | sb))


def _complement_bonus(visual_sim01: float) -> float:
    """Prefer moderate similarity when complementing (peak away from 1.0 and 0.0)."""
    # visual_sim01 in [0,1]; distance peaks around mid-high diversity
    d = 1.0 - visual_sim01
    mu = 0.42
    sigma = 0.18
    return float(np.exp(-((d - mu) ** 2) / (2 * sigma * sigma)))


@dataclass
class ScoreParts:
    visual_style_sim_01: float
    interest_overlap_01: float
    preference_fit_01: float
    activity_visual_overlap_01: float
    complement_style_bonus_01: float
    penalty_01: float
    total_01: float
    total_display: float


def score_pair(query: UserPersona, cand: UserPersona) -> ScoreParts:
    visual_sim01 = _cos01(query.clip_centroid, cand.clip_centroid)
    interest01 = _jaccard(query.interest_tags, cand.interest_tags)
    pref_fit01 = _cos01(query.preference_embedding, cand.clip_centroid)
    act01 = _prompt_overlap_score(query, cand)
    comp_bonus01 = _complement_bonus(visual_sim01)

    penalty = 0.0
    # Mild penalty if candidate lacks any detector cues when query has many YOLO tags (demo robustness)
    if len(query.yolo_tags) >= 4 and len(cand.yolo_tags) == 0:
        penalty += 0.05

    if query.preference_mode == "similar":
        w = dict(v=0.38, i=0.22, p=0.28, a=0.12, c=0.05)
        comp_scaled = comp_bonus01 * 0.35  # down-weight complement peak in similar mode
    else:
        w = dict(v=0.22, i=0.14, p=0.34, a=0.18, c=0.22)
        comp_scaled = comp_bonus01

    raw = (
        w["v"] * visual_sim01
        + w["i"] * interest01
        + w["p"] * pref_fit01
        + w["a"] * act01
        + w["c"] * comp_scaled
        - penalty
    )
    raw = float(np.clip(raw, 0.0, 1.0))
    display = float(100.0 * raw)

    return ScoreParts(
        visual_style_sim_01=visual_sim01,
        interest_overlap_01=interest01,
        preference_fit_01=pref_fit01,
        activity_visual_overlap_01=act01,
        complement_style_bonus_01=comp_bonus01,
        penalty_01=float(penalty),
        total_01=raw,
        total_display=display,
    )


def rank_candidates(query: UserPersona, candidates: dict[str, UserPersona], top_k: int = 10) -> list[tuple[str, ScoreParts]]:
    rows: list[tuple[str, ScoreParts]] = []
    for uid, p in candidates.items():
        if uid == query.user_id:
            continue
        rows.append((uid, score_pair(query, p)))
    rows.sort(key=lambda x: x[1].total_display, reverse=True)
    return rows[:top_k]
