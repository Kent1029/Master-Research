import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse



def main(args):
    # 定義自定義數據集
    class ELA_WISE_Dataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = {'image': self.data[idx], 'label': self.labels[idx]}
            if self.transform:
                sample['image'] = self.transform(sample['image'])
            return sample

    # 定義數據轉換
    transform = transforms.Compose([
        transforms.ToTensor(),  # 將圖像轉換為PyTorch張量
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 將灰度轉為三通道
        transforms.Resize((224, 224),antialias=True),  # 調整圖像大小以匹配EfficientNetV4的預訓練模型
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化圖像
    ])

    # 創建自定義數據集實例
    test_dataset = ELA_WISE_Dataset(img, label, transform=transform)

    # 創建數據加載器
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 载入模型
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    model.load_state_dict(torch.load('ELA_WISE_model.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    if args.gpu == 1:
        model=model.to(device)
        print("Use One GPU",device,)
        # 定義損失函數和優化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    else:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model=model.to(device)
            # 定義損失函數和優化器
            criterion = nn.CrossEntropyLoss()
            #optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            model=model.to('cuda')


    model.eval()
    # 初始化
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    # 推理
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 计算准确率和AUC
    acc = correct / total
    print(f"Acc: {acc}")
    auc = roc_auc_score(true_labels, predicted_labels)
    print(f"AUC: {auc}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,default='all',help='指定dataset')
    parser.add_argument('-g', '--gpu', type=int,default='1',help='多GPU')
    args = parser.parse_args()

    main(args)