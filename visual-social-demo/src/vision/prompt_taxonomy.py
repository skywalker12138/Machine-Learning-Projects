"""CLIP 中文提示词与「场景 / 活动 / 穿搭」多标签分类（用于聚合视觉画像，非用户预填）。"""
from __future__ import annotations

from typing import Final

# (提示词, 所属维度；一条可属于多类，便于题目要求的「场景、活动、穿搭」拆分展示)
VISUAL_PROMPT_SPECS: Final[list[tuple[str, frozenset[str]]]] = [
    ("户外徒步山地场景", frozenset({"scene", "activity"})),
    ("露营帐篷与篝火", frozenset({"scene", "activity"})),
    ("海边冲浪沙滩度假", frozenset({"scene", "activity", "outfit"})),
    ("城市街拍嘻哈宽松穿搭", frozenset({"scene", "activity", "outfit"})),
    ("极简咖啡馆室内阅读", frozenset({"scene", "activity", "outfit"})),
    ("健身房跑步运动穿搭", frozenset({"scene", "activity", "outfit"})),
    ("艺术馆展览夜景城市", frozenset({"scene", "activity"})),
    ("夜店霓虹灯光派对", frozenset({"scene", "activity", "outfit"})),
    ("办公桌程序员键盘屏幕", frozenset({"scene", "activity"})),
    ("居家厨房烘焙料理", frozenset({"scene", "activity"})),
    ("瑜伽冥想绿植室内", frozenset({"scene", "activity", "outfit"})),
    ("摩托车公路旅行", frozenset({"scene", "activity", "outfit"})),
    ("植物园雨后绿意写生", frozenset({"scene", "activity"})),
    ("科幻LED科技感展厅", frozenset({"scene", "activity"})),
    ("公园遛狗宠物互动", frozenset({"scene", "activity"})),
    ("音乐节草地户外演出", frozenset({"scene", "activity", "outfit"})),
    ("滑雪场羽绒服雪景", frozenset({"scene", "activity", "outfit"})),
    ("篮球场球鞋运动场", frozenset({"scene", "activity", "outfit"})),
    ("汉服古镇传统茶道", frozenset({"scene", "activity", "outfit"})),
    ("潜水海岛蓝色海水", frozenset({"scene", "activity", "outfit"})),
    ("电竞RGB桌面电脑椅", frozenset({"scene", "activity", "outfit"})),
    ("登山包装备登山地图", frozenset({"scene", "activity", "outfit"})),
    ("海港码头海鸥工业风", frozenset({"scene", "activity", "outfit"})),
    ("宅家沙发投影仪观影", frozenset({"scene", "activity"})),
    ("陶艺工坊手工艺围裙", frozenset({"scene", "activity", "outfit"})),
    ("夜市街边拥挤烟火气", frozenset({"scene", "activity"})),
    ("原木日式极简客厅", frozenset({"scene", "outfit"})),
    ("正装交响乐音乐厅观众", frozenset({"scene", "activity", "outfit"})),
    ("冲锋衣工装户外穿搭", frozenset({"outfit", "activity"})),
    ("极简黑白灰穿搭", frozenset({"outfit"})),
    ("明亮撞色休闲穿搭", frozenset({"outfit"})),
    ("机能风运动穿搭", frozenset({"outfit", "activity"})),
    ("复古胶片相机手持", frozenset({"activity", "scene"})),
]

PROMPT_BANK_ZH: list[str] = [t for t, _ in VISUAL_PROMPT_SPECS]
PROMPT_TO_BUCKETS: dict[str, frozenset[str]] = {t: c for t, c in VISUAL_PROMPT_SPECS}


def top_in_buckets(
    prompt_scores: dict[str, float],
    bucket: str,
    k: int = 8,
) -> list[tuple[str, float]]:
    """从 CLIP 软标签中取出属于某 bucket 的 TOP-k（分数为聚合后的相似度）。"""
    items: list[tuple[str, float]] = []
    for text, score in prompt_scores.items():
        cats = PROMPT_TO_BUCKETS.get(text)
        if cats and bucket in cats:
            items.append((text, float(score)))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]
