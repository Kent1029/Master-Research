import cv2
from facenet_pytorch import MTCNN
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size=299, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

mtcnn = MTCNN()

image = cv2.imread('frame_0.png')  # 替换成您的图像文件路径

boxes, prob = mtcnn.detect(image)
print('boxes:::',boxes)

if boxes is None:
    print("No 'box' found. Skipping...")
else:
    print('boxes:::', boxes)
    for i, face_info in enumerate(boxes):
        x, y, x_w, y_h = [int(coord) for coord in face_info]  # convert coordinates to integers
        face = image[y:y_h, x:x_w]  # 切割出人脸
        cv2.imwrite(f'face_{i}.jpg', face)  # 将切割出的人脸保存为图像文件
    
