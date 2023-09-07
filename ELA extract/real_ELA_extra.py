
from PIL import Image, ImageChops, ImageEnhance
import os
import glob
from tqdm import tqdm

# feature engineering - Error Level Analysis
def ELA():
        # 載入PNG圖像
        png_image_path = "0.png"
        image = Image.open(png_image_path)

        # 保存為JPEG格式
        jpeg_image_path = "0.png"
        image.save(jpeg_image_path, "JPEG")
        print("jpeg_image_path::",jpeg_image_path)

        # 讀取JPEG圖像
        jpeg_image = Image.open(jpeg_image_path)

        # 計算ELA圖像
        ela_image = ImageChops.difference(image, jpeg_image)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # 保存ELA圖像
        ela_image_path = "0_ela.jpg"
        ela_image.save(ela_image_path)

        print("ELA圖像已保存到", ela_image_path)

def ELA2():
        # 輸入目錄，包含第一子資料夾和第二子資料夾
        #input_directory = f'/home/kent/Baseline_method/CADDM/train_images/original_sequences/youtube/raw/frames'
        input_directory = f'/home/kent/Baseline_method/CADDM/train_images/original_sequences/youtube/raw/frames'
        # 輸出目錄，指定要保存ELA圖像的資料夾
        #output_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/'
        output_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/'
        # 搜索第二子資料夾中的所有image.png文件
        second_child_dirs = glob.glob(os.path.join(input_directory,  "*", "*.png"))
        print("path::",os.path.join(input_directory,  "*", "*.png"))
        print('second_child_dirs::',second_child_dirs)
        counter=0

        # 遍歷第一子資料夾
        for image_file in  tqdm(second_child_dirs, desc='Processing'):
                # 讀取PNG圖像
                image = Image.open(image_file)

                # 保存為JPEG格式
                jpeg_image_path = image_file.replace(".png", ".jpg")
                image.save(jpeg_image_path, "JPEG")

                # 讀取JPEG圖像
                jpeg_image = Image.open(jpeg_image_path)

                # 計算ELA圖像
                ela_image = ImageChops.difference(image, jpeg_image)
                extrema = ela_image.getextrema()
                max_diff = max([ex[1] for ex in extrema])
                scale = 255.0 / max_diff

                ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

                # 創建ELA圖像的新路徑
                # 使用計數器來生成新的檔名
                #ela_image_filename = f'real_image{counter}_ela.jpg'
                ela_image_filename = f'real_image{counter}_ela.jpg'
                #ela_image_filename = os.path.basename(image_file).replace(".png", "_ela.jpg")
                ela_image_path = os.path.join(output_directory, ela_image_filename)

                # 保存ELA圖像到指定資料夾
                ela_image.save(ela_image_path)

                print("ELA圖像已保存到", ela_image_path)
                # 刪掉臨時的JPEG圖像
                os.remove(jpeg_image_path)
                counter += 1


if __name__=='__main__':
        #ELA()
        ELA2()
        