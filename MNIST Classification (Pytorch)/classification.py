import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Preparing Data and Building the Model
# Data processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.137,),(0.3081))
])

# Load MNIST dataset
train_data = torchvision.datasets.MNIST(root='./data', train = True , download = True, transform = transform)
test_data = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = transform)

# Create DataLoader
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)

class MNISTClassifier(nn.Module): # 创建一个继承自Module的类
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # 将每张二维图片展平成一行数值
        self.layers = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = MNISTClassifier().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)


# Training
def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0
    current = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        # Track Progress
        running_loss += loss.item()
        predicted = output.max(1)[1]
        total += target.size(0)
        current += predicted.eq(target).sum().item()

        # Print every 100 batches
        if batch_idx % 100 == 0 and batch_idx != 0:
            avg_loss = running_loss / 100
            accuracy = 100 * current / total
            print(f'[{batch_idx * 64}/{60000}]'
                  f'Loss: {avg_loss:.3f} | Accuracy: {accuracy:.1f}%')
            running_loss = 0

# Evaluation
def evaluation(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data,target = data.to(device), target.to(device)
            output = model(data)
            predicted = output.max(1)[1]
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100 * correct / total



# Al together

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f'\nEpoch:{epoch+1}\n')
    train_epoch(model, train_loader, loss_function, optimizer, device)
    accuracy = evaluation(model, test_loader, device)
    print(f'Test accuracy : {accuracy:.2f}%')