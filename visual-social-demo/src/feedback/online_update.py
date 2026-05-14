"""Simple multiplicative boosts from explicit user feedback."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.match.scoring import ScoreParts
from src.profile.builder import UserPersona


@dataclass
class FeedbackStore:
    """Stores like/dislike adjustments keyed by candidate user_id."""

    boosts: dict[str, float] = field(default_factory=dict)

    def like(self, candidate_id: str, delta: float = 0.12) -> None:
        self.boosts[candidate_id] = self.boosts.get(candidate_id, 1.0) * (1.0 + delta)

    def dislike(self, candidate_id: str, delta: float = 0.18) -> None:
        self.boosts[candidate_id] = self.boosts.get(candidate_id, 1.0) * max(0.2, 1.0 - delta)

    def propagate_like_from_visual(
        self,
        liked: UserPersona,
        all_personas: dict[str, UserPersona],
        cosine_floor: float = 0.92,
        echo_delta: float = 0.06,
    ) -> None:
        """Nudge users visually similar to the liked profile."""
        lv = liked.clip_centroid
        for uid, p in all_personas.items():
            if uid == liked.user_id:
                continue
            sim = float(np.dot(lv, p.clip_centroid))
            if sim >= cosine_floor:
                self.boosts[uid] = self.boosts.get(uid, 1.0) * (1.0 + echo_delta)


def apply_feedback_to_scores(rows: list[tuple[str, ScoreParts]], feedback: FeedbackStore) -> list[tuple[str, ScoreParts]]:
    out: list[tuple[str, ScoreParts]] = []
    for uid, sp in rows:
        mult = feedback.boosts.get(uid, 1.0)
        new_disp = float(np.clip(sp.total_display * mult, 0.0, 100.0))
        new_total01 = new_disp / 100.0
        new_parts = ScoreParts(
            visual_style_sim_01=sp.visual_style_sim_01,
            interest_overlap_01=sp.interest_overlap_01,
            preference_fit_01=sp.preference_fit_01,
            activity_visual_overlap_01=sp.activity_visual_overlap_01,
            complement_style_bonus_01=sp.complement_style_bonus_01,
            penalty_01=sp.penalty_01,
            total_01=new_total01,
            total_display=new_disp,
        )
        out.append((uid, new_parts))
    out.sort(key=lambda x: x[1].total_display, reverse=True)
    return out
