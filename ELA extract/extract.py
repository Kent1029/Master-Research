
from PIL import Image, ImageChops, ImageEnhance
import os
import glob
from tqdm import tqdm
import cv2
import argparse
from face_dlib import get_dlib_face,conservative_crop,get_landmark,crop_region_face,get_region_face_landmark_mask,ELA,element_wise

# feature engineering - Error Level Analysis and ELA mask
def crop_ela_mask(args):
    # 輸入目錄，包含第一子資料夾和第二子資料夾
    #input_directory = f'/home/kent/Baseline_method/CADDM/train_images/original_sequences/youtube/raw/frames'
    input_directory = f'/home/kent/Baseline_method/CADDM/train_images/FF++/manipulated_sequences/{args.dataset}/raw/frames/'
    # 輸出目錄，指定要保存ELA圖像的資料夾
    #output_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/'
    output_directory = f'/home/kent/dataset/ELA_data/manipulated_sequences/{args.dataset}'
    crop_output_path=f'{output_directory}/face/'
    os.makedirs(crop_output_path, exist_ok=True)
    ela_output_path=f'{output_directory}/ela/'
    os.makedirs(ela_output_path, exist_ok=True)
    mask_output_path=f'{output_directory}/mask/'
    os.makedirs(mask_output_path, exist_ok=True)
    element_wise_path=f'{output_directory}/element_wise/'
    os.makedirs(element_wise_path, exist_ok=True)
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
        crop_image_path=crop_region_face(crop_image,crop_landmark,crop_output_path,counter,args.type)   
        ela_image_path=ELA(crop_image_path,ela_output_path,counter,args.type)
        mask_image_path=get_region_face_landmark_mask(ela_image_path,mask_output_path,counter,args.type)
        element_wise(crop_image_path,mask_image_path, element_wise_path,counter,args.type)
        counter += 1

if __name__=='__main__':
    #ELA()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,default='Deepfakes',help='選擇dataset')
    parser.add_argument('-t', '--type', type=str,default='real',help='選擇real or fake 影片')
    args = parser.parse_args()

    crop_ela_mask(args)
        