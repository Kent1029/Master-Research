import cv2 
from PIL import Image, ImageChops, ImageEnhance
import os
import glob
import csv
#from tqdm import tqdm
import argparse

def noise(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('input_image',image)
     # 將圖像轉換為二值化，閾值設為127
    #_, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    # 高通濾波器，例如拉普拉斯濾波器
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    cv2.imshow('laplacian_image',laplacian_abs)
    _, binary_image = cv2.threshold(laplacian_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('binary_image_image',binary_image)
    # 連通元件標記來計算噪點
    num_labels, labels = cv2.connectedComponents(binary_image)
    # 去掉背景連通元件
    num_noise_points = num_labels - 1
    print(f'Number of noise points: {num_noise_points}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return num_noise_points



def main():
    if args.type == 'real':
        input_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/'
    elif args.type == 'fake':
        input_directory = f'/home/kent/dataset/ELA_data/manipulated_sequences/Deepfakes/'
    
    output_directory = f'/home/kent/dataset/Noise/'
    # 搜索第二子資料夾中的所有image.png文件
    second_child_dirs = glob.glob(os.path.join(input_directory, "*.jpg"))
    print("path::",os.path.join(input_directory, "*.jpg"))
    print('second_child_dirs::',second_child_dirs)
    noise_data = []  # 用于存储PSNR和标签的列表

    # 遍歷第一子資料夾
    for image_file in tqdm(second_child_dirs, desc='Processing'):
        noise_count=noise(image_file)
        # 将PSNR和标签添加到列表
        if args.type == 'real':
            noise_data.append([noise_count, 1])
        elif args.type == 'fake':
            noise_data.append([noise_count, 0])        

    # 将PSNR数据写入CSV文件
    if args.type == 'real':
        csv_file_path = os.path.join(output_directory, f"{args.type}_noise_data.csv")
    elif args.type == 'fake':
        csv_file_path = os.path.join(output_directory, f"{args.type}_noise_data.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Noise", "Label"])
        writer.writerows(noise_data)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-t',dest='type',choices=['real','fake'],default='real')
    args=parser.parse_args()
    #main()
    noise('real_ela.jpg')
    noise('fake_ela.jpg')
    
    
    
