
from PIL import Image, ImageChops, ImageEnhance
import os
import glob
from tqdm import tqdm
import cv2
from element_wise import fake_element_wise
from fake_face_dlib import get_dlib_face,conservative_crop,get_landmark,crop_region_face,get_region_face_landmark_mask,ELA

# feature engineering - Error Level Analysis
def old_ELA():
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

def crop_ela_mask():
        # 輸入目錄，包含第一子資料夾和第二子資料夾
        #input_directory = f'/home/kent/Baseline_method/CADDM/train_images/original_sequences/youtube/raw/frames'
        input_directory = f'/home/kent/Baseline_method/CADDM/train_images/FF++/manipulated_sequences/Deepfakes/raw/frames/'
        # 輸出目錄，指定要保存ELA圖像的資料夾
        #output_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/'
        output_directory = f'/home/kent/dataset/ELA_data/manipulated_sequences/Deepfakes'
        crop_output_path=f'{output_directory}/face/'
        ela_output_path=f'{output_directory}/ela/'
        mask_output_path=f'{output_directory}/mask/'
        element_wise_path=f'{output_directory}/element_wise/'
        # 搜索第二子資料夾中的所有image.png文件
        second_child_dirs = glob.glob(os.path.join(input_directory,  "*", "*.png"))
        print("path::",os.path.join(input_directory,  "*", "*.png"))
        print('second_child_dirs::',second_child_dirs)
        counter=0

        # 遍歷資料夾
        for image_file in tqdm(second_child_dirs, desc='Processing'):
            # 讀取PNG圖像
            image=cv2.imread(image_file)

            face = get_dlib_face(image)
            crop_image=conservative_crop(image,face)
            crop_landmark=get_landmark(crop_image)
            crop_image_path=crop_region_face(crop_image,crop_landmark,crop_output_path,counter)
    
            ela_image_path=ELA(crop_image_path,ela_output_path,counter)
            mask_image_path=get_region_face_landmark_mask(ela_image_path,mask_output_path,counter)
            fake_element_wise(crop_image_path,mask_image_path, element_wise_path,counter)
            counter += 1

if __name__=='__main__':
        #ELA()
        crop_ela_mask()
        