# 集装箱智能破损检测

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)

## 项目简介
本项目为 **2025年 MathorCup 数学应用挑战赛（大数据竞赛）赛道A** 的解决方案。项目基于最新的 **YOLOv11** 目标检测框架，实现了对集装箱外表面破损的自动化智能检测。

针对港口图像背景复杂（起重机、天空、地面）以及缺陷尺度不一（如大面积生锈与细微裂缝）等挑战，模型能够精准定位并识别以下三类常见损伤：
- `dent` / 凹陷 (类别 ID: 0)
- `hole` / 破洞 (类别 ID: 1)
- `rusty` / 锈蚀 (类别 ID: 2)

## 核心特性
- **双重任务架构**：同时解决是否存在破损的二分类任务（有损/无损）和具体破损的目标检测与定位任务。
- **高级数据增强**：深度集成 `Albumentations` 图像增强库，实施动态像素级与空间级变换，有效缓解类别不平衡及复杂光照导致的过拟合问题。
- **YOLOv11 架构优化**：利用 SPPF 扩大感受野，C2PSA 注意力机制增强深层特征建模，C3k2 可变卷积提升对复杂边界的提取能力。

## 数据集准备
使用赛题提供的 3713 张数据集，需严格按照 YOLO 标准格式排列：

```text
dataset_dir/
 ├── images/
 │   ├── train/
 │   └── val/
 └── labels/
     ├── train/
     └── val/
```

配套的 `data.yaml` 配置如下，请确保类别顺序与 `.txt` 标签文件中的 ID 严格对应：

```yaml
train: /绝对路径/或者/相对路径/images/train
val: /绝对路径/或者/相对路径/images/val
nc: 3
names: ['dent', 'hole', 'rusty']
```

## 环境依赖
```bash
# 克隆仓库
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
cd your-repo-name

# 安装依赖
pip install -r requirements.txt
# 核心依赖包
pip install ultralytics albumentations opencv-python
```

## 快速开始

**1. 模型训练**
注意：在 Windows 系统下训练，请务必将执行代码包裹在 `if __name__ == '__main__':` 中以避免多进程死锁报错。

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11m.pt") # 选择不同大小的预训练权重，如 n, s, m, l, x
    model.train(
        data="data.yaml",
        epochs=300,
        imgsz=640,
        batch=16,
        project="runs/train",
        name="container_defect"
    )
```

**2. 模型推理与测试**
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的最优权重
    model = YOLO("runs/train/container_defect/weights/best.pt")
    # 执行推理
    results = model.predict(source="path/to/test/images", save=True, conf=0.25)
```

## 模型表现
- **分类任务（有损/无损）**：验证集准确率达 84.26%，精确率 1.0000，F1-Score 达 0.9146。
- **目标检测任务**：全品类 mAP@0.5 达到 0.541，其中破洞 (Hole) 类别特征提取最为理想，mAP@0.5 达 0.649。在置信度阈值为 0.397 时，检测取得了查准率与查全率的最佳平衡 (F1=0.59)。
