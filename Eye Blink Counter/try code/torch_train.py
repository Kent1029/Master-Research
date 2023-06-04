import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

# 假設你的眨眼次數數據是一個Numpy數組，形狀為(num_samples, num_features)，標籤是一個Numpy數組，形狀為(num_samples,)

# 將眨眼次數數據和標籤轉換為PyTorch的張量
train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels).float()
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels).float()

# 構建ResNet50模型
model = resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# 將模型轉換為訓練模式
model.train()

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 設定訓練參數
num_epochs = 10
batch_size = 16

# 訓練模型
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(batch_data)
        
        # 計算損失
        loss = criterion(outputs.squeeze(), batch_labels)
        
        # 反向傳播和優化
        loss.backward()
        optimizer.step()
        
    # 每個epoch打印一次損失
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 將模型轉換為評估模式
model.eval()

# 在測試集上評估模型
with torch.no_grad():
    outputs = model(test_data)
    predicted_labels = (outputs.squeeze() >= 0.5).float()
    accuracy = torch.mean((predicted_labels == test_labels).float())
    print('Test accuracy:', accuracy.item())
