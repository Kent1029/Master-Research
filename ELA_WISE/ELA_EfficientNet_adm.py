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
from tqdm.auto import tqdm
import argparse
from PIL import Image
import model_net


def load_images(data_dir):
    images = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(data_dir, filename)
            #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)# 讀取ELA圖像，假設它們是灰度圖像
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            images.append(image)
    return images


def load_labels(data_dir):
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            if filename.startswith("real_"):
                labels.append(1)  # 真實圖像的標籤為1
            elif filename.startswith("fake_"):
                labels.append(0)  # deepfake圖像的標籤為0
    return labels


def ELA_WISE(args):
    # 定義自定義數據集
    class ELA_WISE_Dataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = Image.fromarray(self.data[idx])
            label = self.labels[idx]
            sample = {'image': image, 'label': label}
            if self.transform:
                sample['image'] = self.transform(sample['image'])
            return sample



    if args.dataset=="all":
        # 載入ELA圖像和標籤
        real_dataset=['original_Deepfakes','original_Face2Face','original_FaceSwap','original_NeuralTextures']
        fake_dataset=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
    elif args.dataset=="Deepfakes":
        real_dataset=['original_Deepfakes']
        fake_dataset=['Deepfakes']
    elif args.dataset=="Face2Face":
        real_dataset=['original_Face2Face']
        fake_dataset=['Face2Face']
    elif args.dataset=="FaceSwap":
        real_dataset=['original_FaceSwap']
        fake_dataset=['FaceSwap']
    elif args.dataset=="NeuralTextures":
        real_dataset=['original_NeuralTextures']
        fake_dataset=['NeuralTextures']

    
    All_real_images = []
    ALL_real_labels=[]
    for data in real_dataset:
        real_data_dir = f'/home/kent/dataset/ELA_data/original_sequences/youtube/{data}/train/element_wise/'
        print("real_data_dir::",real_data_dir)
        real_images = load_images(real_data_dir)
        All_real_images.append(real_images)
        real_labels = load_labels(real_data_dir)
        ALL_real_labels.extend(real_labels)

    #fake_dataset=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
    #fake_dataset=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
    All_fake_images = []
    ALL_fake_labels=[]
    for data in fake_dataset:
        fake_data_dir = f'/home/kent/dataset/ELA_data/manipulated_sequences/{data}/train/element_wise/'
        print("fake_data_dir::",fake_data_dir)
        fake_images = load_images(fake_data_dir)
        All_fake_images.append(fake_images)
        fake_labels = load_labels(fake_data_dir)
        ALL_fake_labels.extend(fake_labels)

    # 將真實和deepfake的ELA圖像和標籤合併
    #ALL_images = All_real_images + All_fake_images
    ALL_images = [item for sublist in All_real_images for item in sublist] + [item for sublist in All_fake_images for item in sublist]

    ALL_labels = ALL_real_labels + ALL_fake_labels


    desired_size = (224, 224)
    resized_ALL_images = []

    for image in ALL_images:
        image_resized = cv2.resize(image, desired_size)
        resized_ALL_images.append(image_resized)

    # 將ELA圖像轉換為灰度圖像
    gray_ALL_images = []

    # for ela_image_resized in resized_ALL_images:
    #     gray_ela_image = cv2.cvtColor(ela_image_resized, cv2.COLOR_BGR2GRAY)
    #     gray_ALL_images.append(gray_ela_image)
        
    # 將ELA圖像轉換為NumPy數組
    #gray_ALL_images = np.array(gray_ALL_images)
    resized_ALL_images=np.array(resized_ALL_images)
    ALL_labels = np.array(ALL_labels)

    X_train=resized_ALL_images
    y_train = ALL_labels


    # 定義數據轉換
    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 將圖像轉換為PyTorch張量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化圖像
    ])

    # 創建自定義數據集實例
    train_dataset = ELA_WISE_Dataset(X_train, y_train, transform=transform)
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 定義EfficientNetV4模型
    #model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)  # 2類別，真實和deepfake
    model=model_net.get(backbone='efficientnet-b4')
    # 訓練模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Use device",device)
    #model=model.to('cuda')
    
    if args.gpu == 1:
        model=model.to(device)
        print("Use One GPU",device)
        # 定義損失函數和優化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=4e-3)
    else:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model=model.to(device)
            model = nn.DataParallel(model)
            # 定義損失函數和優化器
            criterion = nn.CrossEntropyLoss()
            #optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=4e-3)
        else:
            model=model.to('cuda')

    best_auc = 0
    num_epochs = args.epochs
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        predicted_labels_train = []
        true_labels_train = []
        for batch in train_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            locations, confidence,outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            predicted_labels_train.extend(predicted_train.cpu().numpy())
            true_labels_train.extend(labels.cpu().numpy())

        acc_train = correct_train / total_train
        auc_train = roc_auc_score(true_labels_train, predicted_labels_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Acc: {acc_train}, AUC: {auc_train}")
        # Save the model if it has a better AUC than the current best model
        if auc_train > best_auc:
            best_auc = auc_train
            torch.save(model.state_dict(), f'{args.dataset}_adamw_model_save/ELA_WISE_model_{int(round(best_auc, 3)*1000)}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,choices=['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures'],default='all',help='指定dataset')
    parser.add_argument('-e', '--epochs', type=int,default=100,help='輸入epochs數量')
    parser.add_argument('-g', '--gpu', type=int,default=1,help='選擇single or multi GPU')
    parser.add_argument('-gg', '--ggpu', type=int,default=0,help='選擇哪一張 GPU')
    args = parser.parse_args()
    os.makedirs(f'{args.dataset}_adamw_model_save', exist_ok=True)
    if args.ggpu==0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif args.ggpu==1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ELA_WISE(args)
