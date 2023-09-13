
from PIL import Image, ImageChops, ImageEnhance
import os
import pandas as pd
import glob
from tqdm import tqdm
import cv2
import argparse
from face_dlib import get_dlib_face,conservative_crop,get_landmark,crop_region_face,get_region_face_landmark_mask,ELA,element_wise,get_files_from_split

# feature engineering - Error Level Analysis and ELA mask
def crop_ela_mask(args):
    if args.json == 'train':   
        train_split = pd.read_json('train.json', dtype=False)
        train_files_real, train_files_fake = get_files_from_split(train_split)
        real_input_directories = [f'/home/kent/Baseline_method/CADDM/train_images/FF++/original_sequences/youtube/raw/frames/{train_file}' for train_file in train_files_real]
        fake_input_directories = [f'/home/kent/Baseline_method/CADDM/train_images/FF++/manipulated_sequences/{args.dataset}/raw/frames/{train_file}' for train_file in train_files_fake]
        real_output_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/original_{args.dataset}/{args.json}'
        os.makedirs(real_output_directory, exist_ok=True)
        fake_output_directory = f'/home/kent/dataset/ELA_data/manipulated_sequences/{args.dataset}/{args.json}'
        os.makedirs(fake_output_directory, exist_ok=True)
    elif args.json == 'test':
        test_split = pd.read_json('test.json', dtype=False)
        test_files_real, test_files_fake = get_files_from_split(test_split)
        real_input_directories = [f'/home/kent/Baseline_method/CADDM/train_images/FF++/original_sequences/youtube/raw/frames/{test_file}' for test_file in test_files_real]
        fake_input_directories = [f'/home/kent/Baseline_method/CADDM/train_images/FF++/manipulated_sequences/{args.dataset}/raw/frames/{test_file}' for test_file in test_files_fake]
        real_output_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/original_{args.dataset}/{args.json}'
        os.makedirs(real_output_directory, exist_ok=True)
        fake_output_directory = f'/home/kent/dataset/ELA_data/manipulated_sequences/{args.dataset}/{args.json}/'
        os.makedirs(fake_output_directory, exist_ok=True)
    elif args.json == 'val':
        val_split = pd.read_json('val.json', dtype=False)
        val_files_real, val_files_fake = get_files_from_split(val_split)
        real_input_directories = [f'/home/kent/Baseline_method/CADDM/train_images/FF++/original_sequences/youtube/raw/frames/{val_file}' for val_file in val_files_real]
        fake_input_directories = [f'/home/kent/Baseline_method/CADDM/train_images/FF++/manipulated_sequences/{args.dataset}/raw/frames/{val_file}' for val_file in val_files_fake]
        real_output_directory = f'/home/kent/dataset/ELA_data/original_sequences/youtube/original_{args.dataset}/{args.json}'
        os.makedirs(real_output_directory, exist_ok=True)
        fake_output_directory = f'/home/kent/dataset/ELA_data/manipulated_sequences/{args.dataset}/{args.json}/'
        os.makedirs(fake_output_directory, exist_ok=True)

    real_crop_output_path=f'{real_output_directory}/face/'
    fake_crop_output_path=f'{fake_output_directory}/face/'
    os.makedirs(real_crop_output_path, exist_ok=True)
    os.makedirs(fake_crop_output_path, exist_ok=True)

    real_ela_output_path=f'{real_output_directory}/ela/'
    fake_ela_output_path=f'{fake_output_directory}/ela/'
    os.makedirs(real_ela_output_path, exist_ok=True)
    os.makedirs(fake_ela_output_path, exist_ok=True)

    real_mask_output_path=f'{real_output_directory}/mask/'
    fake_mask_output_path=f'{fake_output_directory}/mask/'
    os.makedirs(real_mask_output_path, exist_ok=True)
    os.makedirs(fake_mask_output_path, exist_ok=True)


    real_element_wise_path=f'{real_output_directory}/element_wise/'
    fake_element_wise_path=f'{fake_output_directory}/element_wise/'
    os.makedirs(real_element_wise_path, exist_ok=True)
    os.makedirs(fake_element_wise_path, exist_ok=True)

    # 搜索第二子資料夾中的所有image.png文件
    real_second_child_dirs = [glob.glob(os.path.join(real_input_directory, "*.png")) for real_input_directory in real_input_directories]
    real_second_child_dirs = [path for sub_list in real_second_child_dirs for path in sub_list]
    print("path::",real_second_child_dirs)
    print('real_second_child_dirs::',real_second_child_dirs)
    real_counter=0
    
    # 遍歷資料夾
    for image_file in tqdm(real_second_child_dirs, desc='Processing'):
        # 讀取PNG圖像
        image=cv2.imread(image_file)    
        face = get_dlib_face(image)
        crop_image=conservative_crop(image,face)
        crop_landmark=get_landmark(crop_image)
        type="real"
        crop_image_path=crop_region_face(crop_image,crop_landmark,real_crop_output_path,real_counter,type)   
        ela_image_path=ELA(crop_image_path,real_ela_output_path,real_counter,type)
        mask_image_path=get_region_face_landmark_mask(ela_image_path,real_mask_output_path,real_counter,type)
        element_wise(crop_image_path,mask_image_path, real_element_wise_path,real_counter,type)
        real_counter += 1


    # 搜索第二子資料夾中的所有image.png文件
    fake_second_child_dirs = [glob.glob(os.path.join(fake_input_directory, "*.png")) for fake_input_directory in fake_input_directories]
    fake_second_child_dirs = [path for sub_list in fake_second_child_dirs for path in sub_list]
    print("path::",fake_second_child_dirs)
    print('fake_second_child_dirs::',fake_second_child_dirs)
    fake_counter=0
    
    # 遍歷資料夾
    for image_file in tqdm(fake_second_child_dirs, desc='Processing'):
        # 讀取PNG圖像
        image=cv2.imread(image_file)    
        face = get_dlib_face(image)
        crop_image=conservative_crop(image,face)
        crop_landmark=get_landmark(crop_image)
        type="fake"
        crop_image_path=crop_region_face(crop_image,crop_landmark,fake_crop_output_path,fake_counter,type)   
        ela_image_path=ELA(crop_image_path,fake_ela_output_path,fake_counter,type)
        mask_image_path=get_region_face_landmark_mask(ela_image_path,fake_mask_output_path,fake_counter,type)
        element_wise(crop_image_path,mask_image_path, fake_element_wise_path,fake_counter,type)
        fake_counter += 1

if __name__=='__main__':
    #ELA()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,default='Deepfakes',help='選擇dataset')
    #parser.add_argument('-t', '--type', type=str,default='real',help='選擇real or fake 影片')
    parser.add_argument('-j', '--json', type=str,default='train',help='選擇train or test.json')
    args = parser.parse_args()
    crop_ela_mask(args)