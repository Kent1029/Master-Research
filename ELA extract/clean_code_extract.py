from PIL import Image, ImageChops, ImageEnhance
import os
import pandas as pd
import glob
from tqdm import tqdm
import cv2
import argparse
from face_dlib import (get_dlib_face, conservative_crop, get_landmark, 
                       crop_region_face, get_region_face_landmark_mask,
                       ELA, element_wise, get_files_from_split)

def crop_ela_mask(args):
    if args.json in ['train', 'test', 'val']:
        split = pd.read_json(f'{args.json}.json', dtype=False)
        files_real, files_fake = get_files_from_split(split)
        
        real_input_dirs = [f'/home/kent/Baseline_method/CADDM/train_images/FF++/original_sequences/youtube/raw/frames/{file}' for file in files_real]
        fake_input_dirs = [f'/home/kent/Baseline_method/CADDM/train_images/FF++/manipulated_sequences/{args.dataset}/raw/frames/{file}' for file in files_fake]
        
        real_output_dir = f'/home/kent/dataset/ELA_data/original_sequences/youtube/original_{args.dataset}/{args.json}'
        fake_output_dir = f'/home/kent/dataset/ELA_data/manipulated_sequences/{args.dataset}/{args.json}'
        
        for dir_path in [real_output_dir, fake_output_dir]:
            os.makedirs(dir_path, exist_ok=True)

        paths = {
            "real": {
                "input": real_input_dirs,
                "output": real_output_dir,
                "counter": 0,
                "type": "real"
            },
            "fake": {
                "input": fake_input_dirs,
                "output": fake_output_dir,
                "counter": 0,
                "type": "fake"
            }
        }

        for type, data in paths.items():
            for subdir in ['face', 'ela', 'mask', 'element_wise']:
                os.makedirs(f'{data["output"]}/{subdir}', exist_ok=True)

            second_child_dirs = [glob.glob(os.path.join(dir, "*.png")) for dir in data["input"]]
            second_child_dirs = [path for sublist in second_child_dirs for path in sublist]
            print(f'{type} paths:', second_child_dirs)

            for image_file in tqdm(second_child_dirs, desc=f'Processing {type} images'):
                image = cv2.imread(image_file)
                face = get_dlib_face(image)
                crop_img = conservative_crop(image, face)
                crop_landmark = get_landmark(crop_img)
                crop_img_path = crop_region_face(crop_img, crop_landmark, f'{data["output"]}/face', data["counter"], type)
                ela_img_path = ELA(crop_img_path, f'{data["output"]}/ela', data["counter"], type)
                mask_img_path = get_region_face_landmark_mask(ela_img_path, f'{data["output"]}/mask', data["counter"], type)
                element_wise(crop_img_path, mask_img_path, f'{data["output"]}/element_wise', data["counter"], type)
                data["counter"] += 1

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='Deepfakes', help='選擇資料集')
    parser.add_argument('-j', '--json', type=str, default='train', help='選擇 train, test, 或 val.json')
    args = parser.parse_args()
    crop_ela_mask(args)
