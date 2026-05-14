# 基于视觉风格的社交匹配推荐 Demo

离线可跑的 Python Demo：从用户生活照与标签、自然语言偏好生成**视觉画像**，计算**视觉风格匹配度 / 兴趣线索相似度 / 偏好满足度**，支持相似与互补两种策略，Streamlit 可视化（排行榜、相似度矩阵、标签云）与用户反馈重排。

## 合规说明

- 不进行颜值评分，不输出「恋爱成功率」。
- 不对肤色、种族、身材等敏感属性做价值判断。
- 输出统一使用「视觉风格匹配度」「兴趣线索相似度」「偏好满足度」等表述。

## 环境

- Python 3.10+
- 首次运行会由 Transformers 缓存下载 CLIP 权重（需联网）；YOLOv8 权重由 ultralytics 自动缓存（可选，失败时自动降级为仅 CLIP）。

## 安装

```bash
cd visual-social-demo
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Anaconda / 多环境注意

- **同一个进程里只有一个 Python**。Streamlit 用的是「启动它的那条」`python.exe`，不会自动读你在 IDE 里选的别的环境。
- 若在 conda `pytorch` 里装了 `torch`，但用 **Anaconda base** 里的 `streamlit` 启动，base 里仍会报 `No module named 'torch'`。
- **推荐**：先 `conda activate pytorch`，再在该终端执行 `python -m pip install -r requirements.txt`（必要时）和 `python -m streamlit run app/demo.py`。也可用 **完整路径**：`你的env\python.exe -m streamlit run app\demo.py`。
- 自检当前 Streamlit 使用的解释器：在同一终端运行 `python -c "import sys; print(sys.executable)"`，应与装 torch 的环境一致。

## 准备数据

生成 `data/users.json`（32 个模拟用户）与占位 PNG（无需外网）：

```bash
python scripts/build_users_json.py
python scripts/bootstrap_data.py
```

### 换用真实人脸照（可选，开源 LFW）

占位图为几何色块；若需**真实人脸**做演示，可使用公开数据集 **Labeled Faces in the Wild**，由脚本自动下载并写入各用户的 `img0.png`–`img3.png`（同一用户的 4 张图为**同一身份**的不同照片）：

```bash
python scripts/populate_lfw_faces.py --overwrite
```

首次运行需**联网**，数据说明与引用格式见 [data/DATASETS.md](data/DATASETS.md)。填充人脸后若希望 **昵称用 LFW 身份名、兴趣与偏好仍用内置模拟自述**（符合题目「多用户资料」且与「视觉画像由图生成」分离），请执行：

```bash
python scripts/merge_lfw_stated_profiles.py --backup
```

`lfw_manifest.json` 仅用于数据集溯源；**场景/物体/穿搭/活动**由程序对图片实时分析（见 DATASETS.md）。

若只想用手头照片，直接把文件放到 `data/images/<user_id>/` 下并保持与 `users.json` 中路径一致即可。

## 运行 Demo

必须在项目根目录执行（不要用 `python app/demo.py`，否则会缺少 Streamlit 脚本上下文并报错）：

```bash
streamlit run app/demo.py
```

若出现 `cannot import name 'CLIPModel'`，说明 `transformers` 过旧或安装不完整，请升级：
```bash
pip install -U "transformers>=4.38"
```

若报错 **`torch.load` / CVE-2025-32434 / 要求 torch>=2.6**：新版本 `transformers` 加载 `.bin` 权重时对 PyTorch 版本有下限；本项目已**优先用 safetensors** 加载 CLIP。请先在同一环境中执行：

```bash
python -m pip install -U safetensors "torch>=2.6" torchvision
```

若仍走旧缓存里的 `pytorch_model.bin`，可删掉 Hugging Face 缓存里对应模型目录后重试，或仅升级 torch。

## 项目结构

| 路径 | 说明 |
|------|------|
| `data/users.json` | 模拟用户（≥30） |
| `data/schema.md` | 字段说明 |
| `data/DATASETS.md` | 开源图像数据集说明（含 LFW 引用） |
| `scripts/populate_lfw_faces.py` | 从 LFW 填充真实人脸 PNG |
| `scripts/merge_lfw_stated_profiles.py` | LFW 昵称 + 内置自述合并回 users.json |
| `src/vision/extractor.py` | CLIP / YOLO（可选）/ OCR（可选）逐帧特征 |
| `src/vision/prompt_taxonomy.py` | CLIP 提示词 → 场景/活动/穿搭 分类 |
| `src/profile/builder.py` | 多图聚合画像 |
| `src/match/scoring.py` | 相似 / 互补打分 |
| `src/match/explainer.py` | 可解释理由 |
| `src/feedback/online_update.py` | 反馈更新权重 |
| `app/demo.py` | Streamlit UI |
| `evaluation/case_studies.md` | 5 成功 + 3 失败案例 |

## 命令行快速评测（可选）

```bash
python -m src.cli_recommend --user u001 --top 8
```
