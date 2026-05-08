# PyTorch MNIST Digit Classifier

A clean, efficient, and easy-to-understand implementation of a Feed-Forward Neural Network (Multi-Layer Perceptron) using PyTorch. This project trains a deep learning model to accurately classify handwritten digits (0-9) using the classic MNIST dataset.

## 📖 Overview

This repository demonstrates the fundamental workflow of a PyTorch computer vision project. It covers end-to-end processes including data downloading, tensor transformation, normalization, batching via `DataLoader`, model building, GPU acceleration, and defining custom training/evaluation loops. 

## 🛠️ Tech Stack

- **Framework:** PyTorch (`torch`, `torch.nn`)
- **Computer Vision Library:** `torchvision` (for datasets and transforms)
- **Optimizer:** Adam

## 🧠 Model Architecture

The model (`MNISTClassifier`) is a straightforward Multi-Layer Perceptron (MLP) designed for 28x28 grayscale images:

1. **Flatten Layer:** Converts the 2D image array (28x28) into a 1D continuous tensor of 784 features.
2. **Hidden Layer:** A fully connected (Linear) layer mapping 784 inputs to 128 hidden units.
3. **Activation:** ReLU (Rectified Linear Unit) to introduce non-linearity.
4. **Output Layer:** A fully connected layer mapping the 128 hidden units to 10 output classes (representing the digits 0 through 9).

## 📊 Dataset & Preprocessing

- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) (60,000 training images, 10,000 testing images)
- **Transformations:** - Converted to PyTorch Tensors.
  - Normalized using standard mean and standard deviation `(0.137, 0.3081)` to stabilize and speed up the training process.
- **Batch Size:** 64

## ⚙️ Training Configuration

- **Loss Function:** `CrossEntropyLoss` (ideal for multi-class classification).
- **Optimizer:** Adam with a learning rate of `0.001`.
- **Epochs:** 10
- **Hardware:** Automatically detects and utilizes GPU (`cuda`) if available, otherwise falls back to `cpu`.

## 🚀 How to Run

1. **Clone the repository** (or copy the script into your working directory).
2. **Install the required dependencies:**
   ```bash
   pip install torch torchvision
