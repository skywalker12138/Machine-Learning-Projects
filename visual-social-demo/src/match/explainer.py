"""Template-based explanations (no sensitive attributes)."""
from __future__ import annotations

from src.match.scoring import ScoreParts
from src.profile.builder import UserPersona


def explain_match(query: UserPersona, cand: UserPersona, parts: ScoreParts) -> list[str]:
    lines: list[str] = []

    v_pct = int(round(100 * parts.visual_style_sim_01))
    i_pct = int(round(100 * parts.interest_overlap_01))
    p_pct = int(round(100 * parts.preference_fit_01))
    a_pct = int(round(100 * parts.activity_visual_overlap_01))

    lines.append(f"视觉风格匹配度：整体相近程度约 {v_pct}%（基于图像向量与场景/穿搭线索聚合）。")
    lines.append(f"兴趣线索相似度：标签重合带来的相似度约 {i_pct}%。")
    lines.append(f"偏好满足度：用你的自然语言偏好对齐对方视觉画像的强度约 {p_pct}%。")
    lines.append(f"活动/场景线索重合度（TOP 视觉提示）：约 {a_pct}%。")

    q_top = [t for t, s in query.scene_clues[:2]] + [t for t, s in query.activity_clues[:2]]
    c_top = [t for t, s in cand.scene_clues[:2]] + [t for t, s in cand.activity_clues[:2]]
    q_top = [x for x in q_top if x][:4]
    c_top = [x for x in c_top if x][:4]
    lines.append(f"你在画面中常见的场景/活动线索（CLIP 分类后 TOP）：{'、'.join(q_top) or '（弱信号）'}。")
    lines.append(f"对方画面中常见的场景/活动线索（CLIP 分类后 TOP）：{'、'.join(c_top) or '（弱信号）'}。")

    q_out = [t for t, s in query.outfit_clues[:3]]
    c_out = [t for t, s in cand.outfit_clues[:3]]
    lines.append(f"穿搭风格线索（CLIP）：你方「{'、'.join(q_out) or '—'}」；对方「{'、'.join(c_out) or '—'}」。")

    q_obj = [t.replace("物体:", "") for t, _ in query.object_clues[:4]]
    c_obj = [t.replace("物体:", "") for t, _ in cand.object_clues[:4]]
    lines.append(f"物体检测线索（YOLO，若有）：你方「{'、'.join(q_obj) or '无显著框'}」；对方「{'、'.join(c_obj) or '无显著框'}」。")

    if query.ocr_clues or cand.ocr_clues:
        lines.append(
            "图中文字线索（OCR，可选）："
            f"你方摘录「{'；'.join(query.ocr_clues[:2]) or '无'}」；"
            f"对方「{'；'.join(cand.ocr_clues[:2]) or '无'}」。"
        )

    if query.preference_mode == "complement":
        cb = int(round(100 * parts.complement_style_bonus_01))
        lines.append(f"互补风格加成（可控差异）：该项峰值强度约 {cb}%（不是越高越好，系统偏好适度差异）。")

    if parts.penalty_01 > 1e-6:
        lines.append("稳健性提示：对方缺少部分物体检测线索，分数做了轻微保守下调（Demo 可改进检测管线）。")

    lines.append("说明：分数用于演示排序与可解释性，不代表交友结果预测。")
    return lines


def explain_multimodal_stub(query: UserPersona, cand: UserPersona, parts: ScoreParts) -> str:
    """Reserved hook for local multimodal LLM—returns concise heuristic paragraph."""
    mode_zh = "相似" if query.preference_mode == "similar" else "互补"
    return (
        f"（启发式摘要）在「{mode_zh}」偏好下，候选「{cand.nickname}」与你的"
        f"场景/活动/穿搭 CLIP 线索及 YOLO 物体线索与自述偏好的综合对齐程度较高；"
        f"综合得分约 {parts.total_display:.1f}/100。"
    )
