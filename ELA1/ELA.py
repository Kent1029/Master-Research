import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import csv

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
    #print("extrema",extrema)
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
x=[]
def save_ela_as_csv(ela_image, csv_path):
    # 將ELA圖像轉換為NumPy數組
    ela_array = np.array(ela_image.resize((128,128))).flatten() / 255.0

    ela_array = np.array(ela_array).astype("float64")
    ela_array = ela_array.reshape(-1, 128, 128, 3)
    x.append(ela_array)
    np.save('ela_array.npy',x)
    #print("ela_array",ela_array)


    # 將NumPy數組保存為CSV檔案
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(ela_array.tolist())

def prepare_image(image_path):
    return np.array(ela(image_path, 90).resize((128,128))).flatten() / 255.0


# 使用範例
ela_image_path = "0000.png"
csv_path = "ela_pixels.csv"
ela_image = ela(ela_image_path, quality=90)
ela_image.show()
save_ela_as_csv(ela_image, csv_path)

