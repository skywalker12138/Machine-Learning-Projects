"""Streamlit demo UI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bootstrap_images import ensure_placeholder_images
from src.feedback.online_update import FeedbackStore, apply_feedback_to_scores
from src.io_profiles import load_profiles
from src.match.explainer import explain_match, explain_multimodal_stub
from src.match.scoring import rank_candidates
from src.profile.builder import build_all_personas
from src.vision.extractor import VisionExtractor


def _require_streamlit_runtime() -> None:
    """Streamlit 脚本必须在 `streamlit run` 下执行，否则没有 ScriptRunContext。"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        get_script_run_ctx = None  # type: ignore[assignment]

    if get_script_run_ctx is None or get_script_run_ctx() is None:
        print(
            "\n请使用 Streamlit 启动（在项目根目录 visual-social-demo 下）：\n"
            "  streamlit run app/demo.py\n\n"
            "不要直接使用：python app/demo.py\n"
        )
        raise SystemExit(2)


def _init_session_feedback() -> FeedbackStore:
    if "feedback" not in st.session_state:
        st.session_state.feedback = FeedbackStore()
    return st.session_state.feedback


@st.cache_resource(show_spinner=True)
def _extractor() -> VisionExtractor:
    return VisionExtractor()


def main() -> None:
    _require_streamlit_runtime()
    st.set_page_config(page_title="视觉风格社交推荐 Demo", layout="wide")
    st.title("基于视觉风格的社交匹配推荐 Demo")
    st.caption("离线开源模型（CLIP + 可选 YOLO）。不进行颜值评分，不做恋爱成功率预测。")

    ensure_placeholder_images(ROOT)
    users = load_profiles(ROOT)
    personas = build_all_personas(users, _extractor(), ROOT)
    feedback = _init_session_feedback()

    user_ids = [u["user_id"] for u in users]
    id_to_nick = {u["user_id"]: u["nickname"] for u in users}

    qid = st.sidebar.selectbox("选择查询用户", user_ids, format_func=lambda x: f"{x} · {id_to_nick[x]}")
    top_k = st.sidebar.slider("推荐 Top-K", min_value=3, max_value=20, value=8)
    show_mm = st.sidebar.checkbox("显示启发式多模态摘要（占位，可换本地多模态模型）", value=True)

    if st.sidebar.button("重新加载 CLIP/YOLO 并全量重算画像"):
        st.cache_resource.clear()
        st.rerun()

    query = personas[qid]

    with st.expander("① 用户自述资料（来自 `users.json`，人工/模拟输入，非图像检测）", expanded=False):
        st.markdown(f"**user_id**：{query.user_id}")
        st.markdown(f"**昵称（展示名）**：{query.nickname}")
        st.markdown(f"**偏好模式**：{'相似' if query.preference_mode == 'similar' else '互补'}")
        st.markdown(f"**兴趣标签**：{', '.join(query.interest_tags)}")
        st.markdown(f"**自然语言偏好**：{query.preference_text}")

    with st.expander("② 系统视觉画像（对 `images` 中多图 **实时** 运行 CLIP + 可选 YOLO + 可选 OCR 后聚合）", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**场景线索（CLIP 分类）**")
            for t, s in query.scene_clues[:8]:
                st.caption(f"{t} · {s:.2f}")
            if not query.scene_clues:
                st.caption("（弱信号或模型未就绪）")
        with c2:
            st.markdown("**活动线索（CLIP 分类）**")
            for t, s in query.activity_clues[:8]:
                st.caption(f"{t} · {s:.2f}")
            if not query.activity_clues:
                st.caption("（弱信号）")
        with c3:
            st.markdown("**穿搭风格线索（CLIP）**")
            for t, s in query.outfit_clues[:8]:
                st.caption(f"{t} · {s:.2f}")
            if not query.outfit_clues:
                st.caption("（弱信号）")
        with c4:
            st.markdown("**物体线索（YOLO）**")
            for t, s in query.object_clues[:8]:
                st.caption(f"{t} · {s:.1f}")
            if not query.object_clues:
                st.caption("（无检测框或未安装 ultralytics 权重）")
        st.markdown("**图中文字（OCR，可选）**：安装 [Tesseract](https://github.com/tesseract-ocr/tesseract) 与 `pip install pytesseract` 后自动抽取。")
        if query.ocr_clues:
            st.code("\n".join(query.ocr_clues[:6]), language="text")
        else:
            st.caption("当前未识别到文字或未启用 OCR。")
        st.markdown("**CLIP 原始 TOP 提示（未分类）**：")
        tops = ", ".join([f"{t} ({s:.2f})" for t, s in query.top_visual_prompts[:10]])
        st.caption(tops or "—")

    st.caption(
        "说明：`data/lfw_manifest.json` 仅记录「哪张 LFW 脸填到哪个 user_id」，**不是**视觉分析结果；"
        "上表画像每次启动/重算时由图像流水线生成。"
    )

    rows = rank_candidates(query, personas, top_k=top_k)
    rows_adj = apply_feedback_to_scores(rows, feedback)

    st.subheader("推荐排行榜（含反馈调整后的排序）")
    table_rows = []
    for rank, (cid, sp) in enumerate(rows_adj, start=1):
        table_rows.append(
            {
                "排名": rank,
                "候选": f"{cid} · {personas[cid].nickname}",
                "匹配分数": round(sp.total_display, 2),
                "视觉风格匹配度(0-100)": round(100 * sp.visual_style_sim_01, 2),
                "兴趣线索相似度(0-100)": round(100 * sp.interest_overlap_01, 2),
                "偏好满足度(0-100)": round(100 * sp.preference_fit_01, 2),
                "反馈系数": round(feedback.boosts.get(cid, 1.0), 3),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.subheader("推荐详情与反馈")
    for cid, sp in rows_adj:
        with st.container(border=True):
            hcol1, hcol2, hcol3 = st.columns([3, 1, 1])
            with hcol1:
                st.markdown(f"**{cid} · {personas[cid].nickname}** ｜ 总分 **{sp.total_display:.1f} / 100**")
            with hcol2:
                if st.button("👍 正向反馈", key=f"like-{qid}-{cid}"):
                    feedback.like(cid)
                    feedback.propagate_like_from_visual(personas[cid], personas)
                    st.success("已记录正向反馈，并对相似视觉画像用户施加轻微加成。")
            with hcol3:
                if st.button("👎 负向反馈", key=f"dislike-{qid}-{cid}"):
                    feedback.dislike(cid)
                    st.warning("已记录负向反馈，候选排序系数下调。")

            expl = explain_match(query, personas[cid], sp)
            st.markdown("\n".join([f"- {x}" for x in expl]))
            if show_mm:
                st.info(explain_multimodal_stub(query, personas[cid], sp))

    st.subheader("相似度矩阵（CLIP 聚合向量余弦）")
    ids = user_ids
    mat = np.zeros((len(ids), len(ids)), dtype=np.float32)
    for i, a in enumerate(ids):
        va = personas[a].clip_centroid
        for j, b in enumerate(ids):
            vb = personas[b].clip_centroid
            mat[i, j] = float(np.dot(va, vb))

    fig = px.imshow(
        mat,
        x=[id_to_nick[i] for i in ids],
        y=[id_to_nick[i] for i in ids],
        color_continuous_scale="Blues",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(color="cosine"),
    )
    fig.update_layout(height=760)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("标签云（条形近似：视觉提示 TOP-N）")
    tag_user = st.selectbox("查看用户", ids, format_func=lambda x: f"{x} · {id_to_nick[x]}", key="tag-user")
    p = personas[tag_user]
    df_tags = pd.DataFrame(p.top_visual_prompts, columns=["tag", "weight"]).head(25)
    fig2 = go.Figure(go.Bar(x=df_tags["weight"], y=df_tags["tag"], orientation="h"))
    fig2.update_layout(height=520, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("合规 / 局限"):
        st.markdown(
            "- 分数用于演示排序与解释，不代表社交结果。\n"
            "- 若未安装或加载 YOLO 权重，物体线索可能为空，仍可进行 CLIP 风格分析。\n"
            "- 占位图仅为流程演示；替换为学生授权/公开授权照片可获得更有意义的结果。"
        )


if __name__ == "__main__":
    main()
