from ultralytics import YOLO
import os

if __name__ == '__main__':
    # =========================
    # 1. 数据路径配置
    # =========================
    # 注意路径中要用两个反斜杠 \\ 或者加上 r'' 原始字符串形式
    data_yaml_path = r"D:\mathorcup\A\数据集3713\data.yaml"

    # =========================
    # 2. 自动生成 data.yaml 文件
    # =========================
    data_yaml_content = f"""
train: D:/mathorcup/A/数据集3713/images/train



val: D:/mathorcup/A/数据集3713/images/test   # 验证集暂时用 test 集代替
nc: 3
names: ['dent','hole','rusty']
"""

    os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content.strip())

    print(f"✅ data.yaml 已生成：{data_yaml_path}")

    # =========================
    # 3. 选择 YOLOv11 模型并训练
    # =========================
    # 你可以选择轻量模型 yolo11n.pt 或更强的 yolo11m.pt / yolo11l.pt
    model = YOLO("yolo11n.pt")  # 自动下载模型权重

    # =========================
    # 4. 训练参数设置
    # =========================
    model.train(
        data=data_yaml_path,   # 数据配置文件路径


        epochs=100,            # 训练轮数
        imgsz=640,             # 输入图片尺寸
        batch=16,              # 每批次大小，可根据显存调整
        project="runs/train",  # 结果保存目录
        name="yolo11_hole_detect",  # 当前实验名
        exist_ok=True,         # 允许覆盖
        # workers=4            # 如果加上这个 if __name__ 块后你的电脑还是卡顿或报错，可以尝试解开这行注释，把进程数调小(默认通常是8)
    )

    # =========================
    # 5. 训练完成后进行推理测试
    # =========================
    results = model.predict(
        source=r"D:\mathorcup\A\数据集3713\images\test",  # 测试集路径
        save=True,      # 保存预测结果图片
        imgsz=640
    )

    print("✅ 推理完成！结果已保存在 runs/predict 文件夹中。")