"""CLIP embeddings, optional YOLO tags, CLIP-based style/scene/activity soft labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.vision.prompt_taxonomy import PROMPT_BANK_ZH


@dataclass
class FrameFeatures:
    clip_vec: np.ndarray  # L2-normalized float32 [dim]
    prompt_scores: dict[str, float]  # cosine similarities in [-1,1] scaled to [0,1]
    yolo_tags: list[str]
    ocr_snippets: list[str]  # 可选：图中文字线索（题目要求 OCR；未安装引擎则为空）


class VisionExtractor:
    def __init__(self, device: str | None = None) -> None:
        try:
            import torch
        except ModuleNotFoundError as e:
            import sys

            exe = sys.executable
            raise ModuleNotFoundError(
                "当前 Python 解释器里没有安装 PyTorch（torch）。\n\n"
                "说明：每个终端 / 每次启动 Streamlit 只会用「一个」Python 环境；"
                "你在别的 conda 环境里装了 torch，不等于「base」或其它环境也有。\n\n"
                "解决办法（任选其一）：\n"
                "1) 先激活含 PyTorch 的环境，再启动：\n"
                "   conda activate pytorch\n"
                "   cd C:\\Users\\13728\\visual-social-demo\n"
                "   python -m streamlit run app/demo.py\n\n"
                "2) 用该环境的 python 显式调用（把路径改成你的 env 路径）：\n"
                "   C:\\Users\\13728\\.conda\\envs\\pytorch\\python.exe -m streamlit run app\\demo.py\n\n"
                "3) 在你「实际用来跑 Streamlit」的环境里安装 torch：\n"
                "   python -m pip install torch torchvision\n\n"
                f"当前解释器路径：{exe}"
            ) from e

        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "无法导入 transformers。请先安装或升级：pip install -U \"transformers>=4.38\""
            ) from e

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "openai/clip-vit-base-patch32"
        # 新版 transformers 对 .bin 权重走 torch.load；旧 torch 会触发安全限制（CVE-2025-32434）。
        # 优先 use_safetensors=True，避免走不安全的 pickle 权重路径。

        def _torch_ge_2_6() -> bool:
            import re

            m = re.match(r"(\d+)\.(\d+)", torch.__version__.split("+")[0])
            if not m:
                return False
            major, minor = int(m.group(1)), int(m.group(2))
            return major > 2 or (major == 2 and minor >= 6)

        try:
            self._clip = AutoModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        except Exception as e1:
            if _torch_ge_2_6():
                try:
                    self._clip = AutoModel.from_pretrained(model_name, use_safetensors=False).to(self.device)
                except Exception as e2:
                    raise RuntimeError(
                        "CLIP 加载失败（即便 torch>=2.6）。可尝试：pip install -U safetensors transformers\n"
                        f"详情：{e2}"
                    ) from e2
            else:
                ver = torch.__version__.split("+")[0]
                raise RuntimeError(
                    "加载 CLIP 失败：当前 PyTorch 版本过低，且未能用 safetensors 加载权重。\n\n"
                    "修复（推荐其一）：\n"
                    "A) 升级 PyTorch：python -m pip install -U \"torch>=2.6\" torchvision\n"
                    "B) 安装 safetensors 后删除旧的 HF 缓存再运行（让 Hub 拉取 *.safetensors）：\n"
                    "   python -m pip install -U safetensors\n"
                    "   删除目录类似：%USERPROFILE%\\.cache\\huggingface\\hub\\models--openai--clip-vit-base-patch32\n\n"
                    f"当前 torch：{ver}\n"
                    f"safetensors 加载错误：{e1}"
                ) from e1
        self._proc = AutoProcessor.from_pretrained(model_name)
        self._clip.eval()

        self._yolo = None
        try:
            from ultralytics import YOLO

            self._yolo = YOLO("yolov8n.pt")
        except Exception:
            self._yolo = None

        self._prompt_texts = list(PROMPT_BANK_ZH)
        self._prompt_emb = self._embed_texts(self._prompt_texts)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        inputs = self._proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            t = self._clip.get_text_features(**inputs)
            t = t / t.norm(dim=-1, keepdim=True)
        return t.detach().cpu().numpy().astype(np.float32)

    def _embed_image_pil(self, img: Image.Image) -> np.ndarray:
        torch = self._torch
        inputs = self._proc(images=img, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            v = self._clip.get_image_features(**inputs)
            v = v / v.norm(dim=-1, keepdim=True)
        return v.detach().cpu().numpy().astype(np.float32)[0]

    def _embed_image_path(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return self._embed_image_pil(img)

    def _optional_ocr(self, img: Image.Image) -> list[str]:
        """可选 OCR：安装 Tesseract + pytesseract 后自动抽取图中文字线索。"""
        try:
            import pytesseract
        except ImportError:
            return []
        try:
            txt = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
            if not txt or len(txt) < 2:
                return []
            # 拆成短片段，避免单条过长
            parts = [p.strip() for p in txt.replace("\r", "\n").split("\n") if p.strip()]
            return parts[:5] if parts else [txt[:120]]
        except Exception:
            return []

    def _yolo_top_tags(self, path: Path, top_k: int = 8) -> list[str]:
        if self._yolo is None:
            return []
        try:
            res = self._yolo.predict(str(path), verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                return []
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names
            order = np.argsort(-confs)[:top_k]
            tags: list[str] = []
            for i in order:
                cls_id = int(clss[i])
                name = str(names.get(cls_id, cls_id))
                tags.append(f"物体:{name}")
            return tags
        except Exception:
            return []

    def extract_frame(self, image_path: Path) -> FrameFeatures:
        path = Path(image_path)
        img = Image.open(path).convert("RGB")
        clip_vec = self._embed_image_pil(img)
        raw_sims = self._prompt_emb @ clip_vec  # [P]
        scaled = ((raw_sims.astype(np.float32) + 1.0) / 2.0).clip(0.0, 1.0)
        # strict= 需 Python 3.10+；此处保持与 3.9 兼容
        prompt_scores = {p: float(s) for p, s in zip(self._prompt_texts, scaled.tolist())}
        ytags = self._yolo_top_tags(path)
        ocr = self._optional_ocr(img)
        return FrameFeatures(clip_vec=clip_vec, prompt_scores=prompt_scores, yolo_tags=ytags, ocr_snippets=ocr)

    def embed_text_single(self, text: str) -> np.ndarray:
        return self._embed_texts([text])[0]


def aggregate_prompt_scores(frame_feats: list[FrameFeatures]) -> dict[str, float]:
    if not frame_feats:
        return {}
    keys = frame_feats[0].prompt_scores.keys()
    out: dict[str, float] = {}
    n = len(frame_feats)
    for k in keys:
        out[k] = float(sum(f.prompt_scores[k] for f in frame_feats) / n)
    return out


def top_prompt_tags(prompt_scores: dict[str, float], k: int = 10) -> list[tuple[str, float]]:
    items = sorted(prompt_scores.items(), key=lambda x: x[1], reverse=True)
    return items[:k]


def aggregate_yolo_tags(frame_feats: list[FrameFeatures], max_tags: int = 12) -> list[tuple[str, float]]:
    counts: dict[str, float] = {}
    for f in frame_feats:
        for t in f.yolo_tags:
            counts[t] = counts.get(t, 0.0) + 1.0
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return items[:max_tags]


def aggregate_ocr_snippets(frame_feats: list[FrameFeatures], max_lines: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for f in frame_feats:
        for line in f.ocr_snippets:
            key = line[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append(line[:200])
            if len(out) >= max_lines:
                return out
    return out
