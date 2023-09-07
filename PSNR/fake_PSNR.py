import cv2 
from PIL import Image, ImageChops, ImageEnhance
import os
import glob
import csv
from tqdm import tqdm
import numpy as np

def PSNR(img1, img2):
    if img1.size != img2.size:
        # 重新調整img2的大小為img1的大小
        img2 = img2.resize(img1.size)
    # 将PIL图像对象转换为NumPy数组
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    # 使用 OpenCV 函數計算 PSNR
    psnr = cv2.PSNR(img1_np, img2_np)
    return psnr


def main():
    input_directory = f'/home/kent/Baseline_method/CADDM/train_images/manipulated_sequences/Deepfakes/raw/frames'
    output_directory = f'/home/kent/dataset/PSNR/'
    # 搜索第二子資料夾中的所有image.png文件
    second_child_dirs = glob.glob(os.path.join(input_directory,  "*", "*.png"))
    print("path::",os.path.join(input_directory,  "*", "*.png"))
    print('second_child_dirs::',second_child_dirs)
    counter=0
    psnr_data = []  # 用于存储PSNR和标签的列表

    # 遍歷第一子資料夾
    for image_file in  tqdm(second_child_dirs, desc='Processing'):
        # 讀取PNG圖像
        image = Image.open(image_file)

        # 保存為JPEG格式
        jpeg_image_path = image_file.replace(".png", ".jpg")
        image.save(jpeg_image_path, "JPEG")

        # 讀取JPEG圖像
        jpeg_image = Image.open(jpeg_image_path)
        
        psnr=PSNR(image,jpeg_image)
        os.remove(jpeg_image_path)
        counter += 1

        # 将PSNR和标签添加到列表
        psnr_data.append([psnr, 0])

    # 将PSNR数据写入CSV文件
    csv_file_path = os.path.join(output_directory, "manipulated_sequences_psnr_data.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PSNR", "Label"])
        writer.writerows(psnr_data)
    
def calculate_original_ELA():
    png_input_directory = f'/home/kent/Baseline_method/CADDM/train_images/manipulated_sequences/Deepfakes/raw/frames'
    jpg_input_directory=f'/home/kent/dataset/ELA_data/manipulated_sequences/Deepfakes/'
    output_directory = f'/home/kent/dataset/PSNR/'
    # 搜索第二子資料夾中的所有image.png文件
    png_child_dirs = glob.glob(os.path.join(png_input_directory,  "*", "*.png"))
    jpg_child_dirs = glob.glob(os.path.join(jpg_input_directory, "*.jpg"))
    print('png數量:::',len(png_child_dirs))
    print('jpg數量:::',len(jpg_child_dirs))

    psnr_data = []  # 用于存储PSNR和标签的列表

    # 遍歷第一子資料夾
    for png_image,jpg_image, in  tqdm(zip(png_child_dirs, jpg_child_dirs), desc='Processing'):
        # 讀取PNG圖像
        png_image = Image.open(png_image)
        # 讀取JPG圖像
        jpg_image = Image.open(jpg_image)

        
        psnr=PSNR(png_image,jpg_image)
        #os.remove(jpeg_image_path)
        # 将PSNR和标签添加到列表
        psnr_data.append([psnr, 0])

    # 将PSNR数据写入CSV文件
    csv_file_path = os.path.join(output_directory, "manipulated_sequences_psnr_data.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PSNR", "Label"])
        writer.writerows(psnr_data)

def strip_prefix(filename):
    return filename.replace("real_", "").replace("fake_", "")

def calculate_realELA_fakeELA():
    real_input_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/'
    fake_input_directory=f'/home/kent/dataset/ELA_data/manipulated_sequences/Deepfakes/'
    output_directory = f'/home/kent/dataset/PSNR/'
    # 搜索第二子資料夾中的所有image.png文件
    real_child_dirs = glob.glob(os.path.join(real_input_directory, "*.jpg"))
    fake_child_dirs = glob.glob(os.path.join(fake_input_directory, "*.jpg"))
    real_filenames = {strip_prefix(os.path.basename(path)): path for path in real_child_dirs}
    fake_filenames = {strip_prefix(os.path.basename(path)): path for path in fake_child_dirs}
    common_filenames = set(real_filenames.keys()).intersection(fake_filenames.keys())
    print('real_ELA數量:::',len(real_filenames))
    print('fake_ELA數量:::',len(fake_filenames))

    psnr_data = []  # 用于存储PSNR和标签的列表

    # 遍歷第一子資料夾
    for filename in  tqdm(common_filenames, desc='Processing'):

        real_image_path = real_filenames[filename]
        fake_image_path = fake_filenames[filename]
        real_image = Image.open(real_image_path)
        fake_image = Image.open(fake_image_path)

        
        psnr=PSNR(real_image,fake_image)
        #os.remove(jpeg_image_path)
        # 将PSNR和标签添加到列表
        psnr_data.append([psnr])

    # 将PSNR数据写入CSV文件
    csv_file_path = os.path.join(output_directory, "psnr_data.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PSNR"])
        writer.writerows(psnr_data)

if __name__=='__main__':
    calculate_realELA_fakeELA()