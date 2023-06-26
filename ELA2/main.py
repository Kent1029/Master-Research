
# https://github.com/z1311/Image-Manipulation-Detection
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import csv
import os
import ela
from model import IMDModel



# $ python main.py -p pathname

# -p or --path: Image pathname (required)
#E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Deepfakes\\c40\\images\000_003\\0000.png

def infer_folder(folder_path, model, device):
    for root, dirs, files in os.walk(folder_path):
        for file_name in tqdm(files):
            if file_name.endswith(".jpeg") or file_name.endswith(".png"):
                file_path = os.path.join(root, file_name)
                infer(file_path, model, device)



def infer(img_path, model, device):
    #print("Performing Level 1 analysis...")
    #level1.findMetadata(img_path=img_path)

    print("Starting Error level analysis...")
    ela.ELA(img_path=img_path)

    img = Image.open("temp/ela_img.jpg")
    img = img.resize((128,128))
    img = np.array(img, dtype=np.float32).transpose(2,0,1)/255.0
    img_array = img.flatten()[:16384]
    img_array = img_array.reshape(1, -1)
    #np.savetxt("img_array.csv", img_array, delimiter=",")
    # 读取已有的csv数据
    existing_data = []
    if os.path.isfile("img_array.csv"):
        with open("img_array.csv", 'r') as file:
            reader = csv.reader(file)
            existing_data = list(reader)

    # 将当前图片的文件名和img_array组合成一行数据
    file_row = [img_path, ','.join(map(str, img_array[0]))]

    # 将当前图片的数据追加到现有csv数据中
    existing_data.append(file_row)

    # 将所有数据写入csv文件
    with open("img_array.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(existing_data)



    img = np.expand_dims(img, axis=0)

    out = model(torch.from_numpy(img).to(device=device))
    y_pred = torch.max(out, dim=1)[1]

    print("Prediction:",end=' ')
    print("Authentic真的" if y_pred else "Tampared假的") # auth -> 1 and tp -> 0





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image Manipulation Detection')

    req_args = parser.add_argument_group('Required Args')
    #req_args.add_argument('-p', '--path', type=str, metavar='img_path', dest='img_path', required=True, help='Image Path')
    req_args.add_argument('-p', '--path', type=str, metavar='folder_path', dest='folder_path',default='E:\\Research\\dataset\\FaceForensics++\\original_sequences\\youtube\\c40', required=False, help='Folder Path')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #selecting device
    print("Working on",device)

    model_path = "model/model_c1.pth"
    model = torch.load(model_path)

    #infer(model=model, img_path=args.img_path, device=device)
    infer_folder(args.folder_path, model, device)


