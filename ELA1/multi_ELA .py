import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import csv
from tqdm import tqdm

def ela(image_path, quality=90):
    # 加載原始圖像
    original_image = Image.open(image_path).convert('RGB')

    # 將原始圖像保存為JPEG格式
    temp_path = "temp.jpg"
    original_image.save(temp_path, "JPEG", quality=quality)

    # 加載重新壓縮後的圖像
    recompressed_image = Image.open(temp_path)

    # 計算ELA圖像
    ela_image = ImageChops.difference(original_image, recompressed_image)
    #ela_image = ela_image.convert("L")

    # 計算ELA圖像的最大差異值
    extrema = ela_image.getextrema()
    max_diff = sum([ex[1] for ex in extrema]) / 3
    if max_diff == 0:
        max_diff = 1

    # 調整ELA圖像的亮度
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # 刪除臨時圖像
    recompressed_image.close()
    original_image.close()
    Image.Image.close(original_image)
    Image.Image.close(recompressed_image)

    return ela_image

def save_ela_as_csv(ela_images, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for ela_image in ela_images:
            ela_array = np.array(ela_image.resize((128, 128))).flatten() / 255.0
            writer.writerow(ela_array.tolist())

# 資料夾路徑
folder_path = "E:\\Research\\dataset\\FaceForensics++\\original_sequences\\youtube\\c40\\images"
output_dir = ".\\data\\youtube\\c40"

# 使用範例
for root, dirs, files in os.walk(folder_path):
    for dir_name in dirs:
        current_folder_path = os.path.join(root, dir_name)
        ela_images = []
        for filename in tqdm(os.listdir(current_folder_path), desc=dir_name):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(current_folder_path, filename)

                ela_image = ela(image_path, quality=90)
                ela_images.append(ela_image)

        # 保存為Numpy檔案
        ela_array = np.array([np.array(ela_image) for ela_image in ela_images])
        ela_array = ela_array.astype("float64")
        ela_array = ela_array.reshape(-1, 128, 128, 3)
        npy_filename = os.path.join(output_dir, dir_name + ".npy")
        np.save(npy_filename, ela_array)

        # 保存為CSV檔案
        csv_filename = os.path.join(output_dir, dir_name + ".csv")
        save_ela_as_csv(ela_images, csv_filename)

        # 釋放已使用的陣列
        ela_images.clear()
        del ela_array
        