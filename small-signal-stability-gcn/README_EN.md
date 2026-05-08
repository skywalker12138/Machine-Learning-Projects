# ResGCN for Power System Small Signal Stability Prediction

This project implements an enhanced Graph Convolutional Network with Residual connections (ResGCN) using PyTorch Geometric, specifically designed for fast regression of Small Signal Stability (SSS) indices.

## 1. Overview: What is Small Signal Stability?
**Small Signal Stability (SSS)** refers to the ability of a power system to maintain synchronism after being subjected to small disturbances, such as continuous minor fluctuations in load or slight changes in line impedance.
* **Traditional Approach:** Relies on eigenvalue analysis of the state-space matrix. As power grids grow in scale and integrate high levels of renewable energy, the computational burden grows exponentially, making real-time monitoring difficult.
* **Project Significance:** By leveraging the fitting power of deep learning, this project establishes a mapping between grid topology and stability indices, enabling millisecond-level stability assessment.

## 2. Motivation: Why Graph Convolutional Networks (GCN)?
### 2.1 Topology-Awareness
Power grids are inherently physical graphs (buses as nodes, lines as edges). Conventional neural networks (like MLP) treat grid parameters as independent vectors, ignoring physical connectivity. GCNs operate directly on graph structures, capturing the mutual coupling between power electronic devices.

### 2.2 Message Passing Mechanism
The propagation of small disturbances exhibits strong spatial correlation. Through the "Message Passing" mechanism, GCNs allow each node to aggregate states from its neighbors, learning both global and local dynamic features.

### 2.3 Advantages of Residual Learning
Deep GNNs are prone to "Over-smoothing," where node features become indistinguishable. This project introduces Residual Connections, allowing raw features to bypass layers. This ensures the model retains low-level physical features while extracting high-level abstract representations.

## 3. Detailed Structural Implementation
The model utilizes a **ResNet-style GCN** architecture, consisting of the following layers:

* **Linear Input Mapping:** Since the raw feature dimension may differ from the hidden dimension, a linear layer `self.lin_in` first projects node features into a higher-dimensional space (128-dim), serving as the baseline for residual additions.
* **3-Layer GCN Residual Blocks:**
    Consists of three consecutive `GCNConv` layers. The output of each layer is passed through a ReLU activation and performed with **Vector Addition (Residual Connection)** with the previous layer's input. This facilitates gradient flow and enables the training of deeper architectures.
* **Global Mean Pooling:**
    To transition from node-level features to a graph-level representation, a pooling operation averages the hidden states of all nodes, forming a feature vector that represents the state of the entire power grid.
* **MLP Decoder (Multi-Layer Perceptron):**
    Comprises 4 fully connected layers (128 -> 64 -> 32 -> Output Dim).
    * **Dropout (0.5):** Randomly deactivates 50% of neurons during training, significantly enhancing generalization and preventing overfitting on small datasets.
    * **ReLU Activation:** Maintained between layers to ensure non-linear expressive power.

## 4. Experimental Results
Performance on the unseen test set (Original Scale):
* **RMSE:** 0.0032
* **MAAPE:** 3.67%
* **SMAPE:** 3.72%

## 5. Installation & Usage
* Requirements: Python 3.x / PyTorch / PyTorch Geometric
* Install: `pip install torch torch_geometric pandas numpy scikit-learn`
* Run: `python main.py`