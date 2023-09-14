import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score
from ELA_EfficientNet import load_images,load_labels
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
        real_data_dir = f'/home/kent/dataset/ELA_data/original_sequences/youtube/{data}/test/element_wise/'
        print("real_data_dir::",real_data_dir)
        real_images = load_images(real_data_dir)
        All_real_images.append(real_images)
        real_labels = load_labels(real_data_dir)
        ALL_real_labels.extend(real_labels)

    #fake_dataset=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
    fake_dataset=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
    All_fake_images = []
    ALL_fake_labels=[]
    for data in fake_dataset:
        fake_data_dir = f'/home/kent/dataset/ELA_data/manipulated_sequences/{data}/test/element_wise/'
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

    X_test=resized_ALL_images
    y_test = ALL_labels


    # 定義數據轉換
    transform = transforms.Compose([
        transforms.ToTensor(),  # 將圖像轉換為PyTorch張量
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 將灰度轉為三通道
        transforms.Resize((224, 224),antialias=True),  # 調整圖像大小以匹配EfficientNetV4的預訓練模型
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化圖像
    ])

    # 創建自定義數據集實例
    test_dataset = ELA_WISE_Dataset(X_test, y_test, transform=transform)
    
    # 創建數據加載器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 载入模型
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    model.load_state_dict(torch.load(f"/home/kent/Baseline_method/ELA_WISE/model_save/ELA_WISE_model_997.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    if args.gpu == 1:
        model=model.to(device)
        print("Use One GPU",device,)
        
    else:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            model=model.to(device)
        else:
            model=model.to('cuda')


    model.eval()
    # 初始化
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    # 推理模型
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
    parser.add_argument('-d', '--dataset', type=str,choices=['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures'],default='all',help='指定dataset')
    parser.add_argument('-g', '--gpu', type=int,default='1',help='多GPU')
    args = parser.parse_args()
    main(args)