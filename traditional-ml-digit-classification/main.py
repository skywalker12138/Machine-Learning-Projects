import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 任务 1：数据准备 & 任务 3：特征表示
# ==========================================
print("--- 任务 1 & 3：数据准备与特征表示 ---")
# 加载 sklearn 自带的 digits 数据集
digits = datasets.load_digits()

# 提取特征数据 (X) 和标签 (y)
X = digits.data  # 已经展平为 64 维的特征向量
y = digits.target  # 真实类别标签 (0-9)
images = digits.images  # 原始 8x8 图像数据

print(f"数据集中图像的总数量: {len(X)}")
print(f"单张原始图像的大小: {images[0].shape}")
print(f"转换后的特征向量维度: {X[0].shape}")
print(f"类别标签范围: {np.unique(y)}")

# ==================== 修改开始 ====================
# 设置想要展示的行数和列数
num_rows = 4
num_cols = 5
total_samples = num_rows * num_cols  # 共 20 个样本

# 稍微调大 figsize 以适应更多图片 (宽度 12, 高度 8)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# 注意这里的切片变成了 [:total_samples]，也就是取前 20 张图片
for ax, img, label in zip(axes.ravel(), images[:total_samples], y[:total_samples]):
    ax.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.suptitle(f"Task 1: {total_samples} Sample Images and Labels", fontsize=16)
plt.tight_layout()
plt.savefig("task1_samples.png", dpi=300, bbox_inches='tight')  # 保存为图片
print(f"已保存图片: task1_samples.png (包含 {total_samples} 个样本)")
# ==================== 修改结束 ====================

# ==========================================
# 任务 2：数据划分
# ==========================================
print("\n--- 任务 2：数据划分 ---")
# 按照 25% 的比例划分测试集，random_state 保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# ==========================================
# 任务 4：模型训练
# ==========================================
print("\n--- 任务 4：模型训练 ---")
# 初始化 6 种传统机器学习模型
models = {
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000),  # 增加迭代次数防止不收敛
    "SVM": SVC(kernel='rbf'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 用于保存每个模型的准确率
results = {}

# ==========================================
# 任务 5：结果比较
# ==========================================
print("\n--- 任务 5：结果比较 ---")
print(f"{'模型名称':<25} | {'测试准确率':<10}")
print("-" * 40)

for name, model in models.items():
    # 1. 训练模型
    model.fit(X_train, y_train)
    # 2. 在测试集上预测
    y_pred = model.predict(X_test)
    # 3. 记录并打印准确率
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:<25} | {acc:.4f}")

# 找出准确率最高和最低的模型
best_model_name = max(results, key=results.get)
worst_model_name = min(results, key=results.get)
print(f"\n准确率最高的模型: {best_model_name} ({results[best_model_name]:.4f})")
print(f"准确率最低的模型: {worst_model_name} ({results[worst_model_name]:.4f})")

# ==========================================
# 任务 6：错误样本分析 (以表现较好的 SVM 为例)
# ==========================================
print("\n--- 任务 6：错误样本分析 ---")
# 我们选择 SVM 作为表现较好的模型进行详细分析
best_model = models["SVM"]
y_pred_best = best_model.predict(X_test)

# 1. 绘制并保存混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Task 6: Confusion Matrix (SVM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("task6_confusion_matrix.png", dpi=300, bbox_inches='tight')  # 改为保存图片
print("已保存图片: task6_confusion_matrix.png")

# 2. 找出被错误分类的样本
errors = np.where(y_test != y_pred_best)[0]
print(f"SVM 模型共分类错误 {len(errors)} 个样本。")

# 显示并保存前 5 个错误分类的样本
if len(errors) > 0:
    fig, axes = plt.subplots(1, min(5, len(errors)), figsize=(12, 3))
    if len(errors) == 1:
        axes = [axes]
    for ax, err_idx in zip(axes, errors[:5]):
        # 注意：要将 64 维向量重新 reshape 回 8x8 才能显示图像
        img_err = X_test[err_idx].reshape(8, 8)
        true_lbl = y_test[err_idx]
        pred_lbl = y_pred_best[err_idx]

        ax.imshow(img_err, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color="red")
        ax.axis('off')
    plt.suptitle("Task 6: Misclassified Samples (SVM)")
    plt.tight_layout()
    plt.savefig("task6_errors.png", dpi=300, bbox_inches='tight')  # 改为保存图片
    print("已保存图片: task6_errors.png")

print("\n代码运行完毕，请在当前目录下查看生成的 .png 图片！")