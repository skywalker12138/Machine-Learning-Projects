import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

# --------------------
# 数据处理
# --------------------

# 节点特征文件路径
node_features_path = r"C:\Users\13728\Desktop\feature_processed1.xlsx"
# 邻接矩阵文件路径
adj_matrix_path = r"C:\Users\13728\Desktop\daona(nomalization).xlsx"
# 标签文件路径
labels_path = r"C:\Users\13728\Desktop\label.xlsx"

# 检查文件是否存在
for file_path in [node_features_path, adj_matrix_path, labels_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到：{file_path}，请检查文件路径和名称。")

# 读取节点特征和邻接矩阵
node_features = pd.read_excel(node_features_path, header=None).values  # [样本数, 节点数 * 特征数]
adj_matrix = pd.read_excel(adj_matrix_path, header=None).values  # [节点数, 节点数]
labels = pd.read_excel(labels_path, header=None).values  # [样本数, 标签数]
#head=None 即不读取标题行

# 获取样本数和节点数
num_samples = node_features.shape[0]
num_nodes = adj_matrix.shape[0]
num_node_features = int(node_features.shape[1] / num_nodes)
num_labels = labels.shape[1]
#shape[0] 表示行数 shape[1]表示列数

# 将节点特征重塑为 [样本数, 节点数, 特征数]
node_features = node_features.reshape(num_samples, num_nodes, num_node_features)

# 标签标准化处理
labels_normalized = 1 / (1 + np.exp(-1 * labels))
#这一行通过Sigmoid函数（也称为逻辑函数）对 labels 数据进行归一化。Sigmoid 函数的表达式是:      在这里，将 labels 中的每个数值作为输入，通过 1 / (1 + np.exp(-1 * labels)) 将原始标签映射到 0 和 1 之间。这种标准化有助于将数据归一化到固定范围内，更适合用于神经网络的训练。
labels_normalized = torch.tensor(labels_normalized, dtype=torch.float32)
#这一行将 labels_normalized 转换为 PyTorch 的张量格式，同时指定数据类型为 torch.float32，适合用于深度学习模型的训练。

# 构建图数据
def create_graph_data(node_features_sample, adj_matrix, label_sample):
#定义了一个函数 create_graph_data，参数包括一个样本的节点特征 node_features_sample，邻接矩阵 adj_matrix，以及标签 label_sample。
    data = Data()
#创建一个 Data 实例 data，这是 PyTorch Geometric 的数据结构，用于存储图的节点特征、边信息和标签等内容。
    data.x = torch.tensor(node_features_sample, dtype=torch.float32)  # [节点数, 特征数]
#将节点特征 node_features_sample 转换为 PyTorch 张量，数据类型为 float32，并赋给 data.x。data.x 的形状为 [节点数, 特征数]，即每个节点对应的特征。
    edge_index = []
#初始化 edge_index 列表，用于存储边的连接信息。
    num_nodes = adj_matrix.shape[0]
#获取节点数量 num_nodes，对应邻接矩阵的行数。
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0 and i != j:
                edge_index.append([i, j])
#通过双层循环遍历邻接矩阵的每个元素。如果 adj_matrix[i, j] != 0 且 i != j，表示节点 i 和 j 之间存在一条边，将其添加到 edge_index 列表中。

    # 如果没有边，避免张量为空
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
#如果 edge_index 为空，则创建一个空张量以避免错误。否则，将 edge_index 转换为 PyTorch 张量，并使用 .t() 转置，使其形状变为 [2, E]（2 表示边的起点和终点，E 为边数）
    data.edge_index = edge_index
    data.y = torch.tensor(label_sample, dtype=torch.float32).reshape(-1, num_labels)  # [1, 标签数]
#将 edge_index 和标签数据 label_sample 分别赋值给 data.edge_index 和 data.y。标签张量 data.y 被转换为二维张量，形状为 [1, 标签数]
    return data
#返回包含节点特征、边信息和标签信息的 data 对象。

# 创建图列表
graphs = []
for i in range(num_samples):
    graph = create_graph_data(
        node_features_sample=node_features[i],
        adj_matrix=adj_matrix,
        label_sample=labels_normalized[i]
    )
    graphs.append(graph)
#graphs 是一个列表，用于存储所有图样本的图数据。通过循环 num_samples 次，逐个将节点特征、邻接矩阵、标签传入 create_graph_data 函数，生成一个 Data 对象。将生成的图 graph 添加到 graphs 列表中。

# 划分数据集 (80%训练集, 10%验证集, 10%测试集)
train_graphs, temp_graphs = train_test_split(graphs, test_size=0.20, random_state=42)
val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.50, random_state=42)
#使用 train_test_split 将 graphs 列表划分为训练集 train_graphs 和临时数据集 temp_graphs，其中 train_test_split 参数 test_size=0.20 表示取 20% 数据作为临时集（包含验证和测试集）。再次使用 train_test_split 将 temp_graphs 划分为验证集 val_graphs 和测试集 test_graphs，每个占临时集的 50%，即占总数据的 10%。

# 创建数据加载器
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
#train_loader、val_loader 和 test_loader 分别为训练、验证和测试数据集的加载器。每个数据加载器将相应的图数据以批量大小为 16 (batch_size=16) 提供给模型。shuffle=True 使得训练集在每个 epoch 中随机打乱，增强模型泛化能力；验证集和测试集不打乱。


# --------------------
# 模型定义（ResNet风格的GCN）
# --------------------
class ResGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(ResGCNModel, self).__init__()
#ResGCNModel 继承自 nn.Module 类，是 PyTorch 中定义模型的标准方式。in_channels 是输入特征的维度，hidden_channels 是隐藏层的维度大小，out_channels 是输出维度，默认为 1。调用 super() 初始化父类。

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
#conv1、conv2、conv3 是三个图卷积层（GCNConv），依次处理输入数据、隐藏层数据，实现信息的逐层传递。
        # 在输入和隐藏层之间添加线性转换，以匹配维度
        self.lin_in = nn.Linear(in_channels, hidden_channels)
#lin_in 是一个线性层，用于将输入特征映射到隐藏层维度，这一步是在输入与第一层 GCN 卷积之间的维度匹配处理。
        # 全连接层
        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, out_channels)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
#定义了四个全连接层（fc1 到 fc4），用于图卷积层的输出进行进一步的特征抽象与压缩。
#dropout 用于防止过拟合。relu 是激活函数，应用于卷积层和全连接层的输出，增加模型的非线性表达能力。
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
#forward 定义了模型的前向传播过程，data.x 是节点特征，data.edge_index 是图的边连接信息。

        # 输入映射到隐藏维度
        x_in = self.lin_in(x)
#x_in 将输入特征通过 lin_in 进行初步的线性转换，为后续的残差连接做好准备。
        # 第一层
        x1 = self.conv1(x, edge_index)
        x1 = self.relu(x1)
        x1 = x1 + x_in  # 残差连接
#第一层图卷积操作将输入数据 x 通过 conv1 和 ReLU 激活后，与初始线性转换的结果 x_in 相加形成残差连接。
        # 第二层
        x2 = self.conv2(x1, edge_index)
        x2 = self.relu(x2)
        x2 = x2 + x1  # 残差连接
#第二层图卷积操作，处理第一层的输出 x1，并与 x1 相加
        # 第三层
        x3 = self.conv3(x2, edge_index)
        x3 = self.relu(x3)
        x3 = x3 + x2  # 残差连接
#第三层图卷积操作，处理第二层的输出 x2，并与 x2 相加。
        # 图级别的表示
        x_pooled = global_mean_pool(x3, data.batch)
#global_mean_pool 用于对图卷积后的结果 x3 进行全局池化，以图为单位汇总特征。
        # 全连接层，加入ReLU激活函数和Dropout层
        x_fc = self.relu(self.fc1(x_pooled))
        x_fc = self.dropout(x_fc)
        x_fc = self.relu(self.fc2(x_fc))
        x_fc = self.dropout(x_fc)
        x_fc = self.relu(self.fc3(x_fc))
        out = self.fc4(x_fc)
#逐层通过全连接层，应用 ReLU 激活和 Dropout，逐渐将特征维度压缩至输出维度。out 为最终输出，用于模型的预测。
        return out

# --------------------
# 模型训练
# --------------------

# 创建ResGCN模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#检查当前是否有 GPU 可用。若有，则设置为 cuda，否则使用 CPU (cpu)。device 是后续模型、数据放置的设备对象，确保代码在 GPU 上运行以加速计算（若可用）。
model = ResGCNModel(in_channels=num_node_features, hidden_channels=128, out_channels=num_labels).to(device)
#创建 ResGCNModel 模型实例，指定输入特征、隐藏层和输出维度，并将模型放置到 device（GPU 或 CPU）上。

# 设置优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
#optimizer：使用 AdamW 优化器，用于更新模型参数。lr 是学习率，weight_decay 是权重衰减系数。
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
#scheduler：使用 ReduceLROnPlateau 学习率调度器，当验证损失连续 5 个 epoch 不下降时，将学习率减少到原来的 10%。
criterion = nn.MSELoss()
#riterion：定义损失函数为均方误差损失（MSELoss），适用于回归任务。


# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
#num_epochs 决定训练的轮数。在每个 epoch 内部，模型设为训练模式 model.train()，并初始化本轮的 total_loss 累计值
    for data in train_loader:
        data = data.to(device)
        #迭代 train_loader 中的每个批次数据，批次数据 data 被移动到 device。
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.to(device))
        #optimizer.zero_grad() 重置梯度，model(data) 前向传播，得到预测 out，再通过损失函数计算 loss

        loss.backward()
        optimizer.step()
        #loss.backward() 计算梯度，optimizer.step() 更新模型参数
        total_loss += loss.item()
        #将本批次损失累加到 total_loss 中，用于记录该 epoch 的总训练损失。

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss:.4f}")

    # 更新学习率调度器
    scheduler.step(total_loss)
#更新学习率调度器，将当前 total_loss 传入。若训练损失不下降，调度器会降低学习率。

    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.to(device))
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}")
#进入验证阶段，model.eval() 设定模型为评估模式（禁用 Dropout 等）。使用 torch.no_grad() 禁止梯度计算，加速推理并节省内存。迭代验证集 val_loader，计算预测 out 和验证损失 val_loss，最后打印验证损失。


# 测试模型
model.eval()
#odel.eval()：将模型设置为评估模式，以禁用 Dropout 等训练模式专属的层。
test_loss = 0
with torch.no_grad():
    #torch.no_grad()：在此环境下禁用梯度计算，从而节省内存，加速推理。
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y.to(device))
        test_loss += loss.item()
#迭代 test_loader，对每批次数据进行预测并计算损失 loss，累加到 test_loss，最后输出测试集的总损失。
print(f"Test Loss: {test_loss:.4f}")

# 评估指标计算
def mean_arctangent_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.arctan(np.abs((y_true - y_pred) / (y_true + 1e-15)))) * 100
#MAAPE (Mean Arctangent Absolute Percentage Error) 计算预测值和真实值之间的角度误差，这种方法可以更有效地处理极小误差的影响。


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
#RMSE (Root Mean Squared Error) 计算均方根误差，用于衡量预测值和真实值之间的误差，越小则代表模型拟合效果越好。

def symmetric_mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape = np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100
    return smape
#SMAPE (Symmetric Mean Absolute Percentage Error) 计算对称平均绝对百分比误差，避免极值误差偏大。epsilon 防止除零错误


# 反归一化函数
def invert_normalization(normalized_values, scale=1, epsilon=1e-15):
    normalized_values = np.clip(normalized_values, epsilon, 1 - epsilon)
    result = -np.log(1 / normalized_values - 1) / scale
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
#invert_normalization：将标准化的预测值和真实值恢复至原始比例。normalized_values 使用 np.clip 限制在 [epsilon, 1 - epsilon] 之间，避免极端值。通过反转归一化公式 -log(1 / normalized_values - 1) / scale 进行反归一化。np.nan_to_num 将结果中的 NaN 或无穷值替换为 0。

# 测试评估
labels_test_numpy = []
predictions_numpy = []
#初始化空列表 labels_test_numpy 和 predictions_numpy 用于存储真实标签和预测值。

with torch.no_grad():
    #torch.no_grad() 禁止梯度计算，加速推理并节省内存。
    for data in test_loader:
        data = data.to(device)
        predictions = model(data)
        labels_test_numpy.append(data.y.cpu().numpy())
        predictions_numpy.append(predictions.cpu().numpy())
#迭代 test_loader 中的每个数据批次：将 data 移动到设备（GPU 或 CPU）上。通过模型预测得到 predictions，并将预测值和真实标签转换为 NumPy 数组格式，追加到相应的列表中。

# 转换为 NumPy 数组
labels_test_numpy = np.vstack(labels_test_numpy)
predictions_numpy = np.vstack(predictions_numpy)
#将每批次的预测值和标签值堆叠成一个完整的二维 NumPy 数组。

# 反归一化预测值和标签
predictions_original_scale = invert_normalization(predictions_numpy)
labels_test_original_scale = invert_normalization(labels_test_numpy)
#使用 invert_normalization 函数将归一化的预测值和真实值还原到原始尺度。

# 计算评估指标
test_maape_original = mean_arctangent_absolute_percentage_error(labels_test_original_scale, predictions_original_scale)
test_rmse_original = root_mean_squared_error(labels_test_original_scale, predictions_original_scale)
test_smape_original = symmetric_mean_absolute_percentage_error(labels_test_original_scale, predictions_original_scale)
#计算整体 MAAPE、RMSE 和 SMAPE，并打印测试集上的总评估指标。

print(
    f'Test MAAPE (Original Scale): {test_maape_original:.2f}%, Test RMSE (Original Scale): {test_rmse_original:.4f}, Test SMAPE (Original Scale): {test_smape_original:.2f}%')

# 打印每个标签的评估指标
for i in range(labels_test_original_scale.shape[1]):  # 标签数
    maape_original = mean_arctangent_absolute_percentage_error(labels_test_original_scale[:, i],
                                                               predictions_original_scale[:, i])
    rmse_original = root_mean_squared_error(labels_test_original_scale[:, i], predictions_original_scale[:, i])
    smape_original = symmetric_mean_absolute_percentage_error(labels_test_original_scale[:, i],
                                                              predictions_original_scale[:, i])
    print(
        f'Tag {i + 1} (Original Scale) MAAPE: {maape_original:.2f}%, RMSE: {rmse_original:.4f}, SMAPE: {smape_original:.2f}%')
#针对每个标签维度分别计算 MAAPE、RMSE 和 SMAPE。通过循环 i，逐个标签打印对应的评估指标，以评估模型对每个标签的预测表现。
