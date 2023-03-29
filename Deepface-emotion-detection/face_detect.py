# deep库的导入就一行代码
from deepface import DeepFace
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import numpy as np
def show_img(imgs: list, img_names: list) -> None:
    imgs_count = len(imgs)
    for i in range(imgs_count):
        ax = plt.subplot(1, imgs_count, i+1) #生成一個子圖，將目前for loop跑到的images顯示在這個子圖中
        ax.imshow(imgs[i])
        ax.set_title(img_names[i])
        ax.set_xticks([]) # 將子圖的橫縱坐標軸刻度設為空，以去除子圖周圍的空白邊框。
        ax.set_yticks([])
    plt.tight_layout(h_pad=3)
    plt.show()

result = DeepFace.detectFace(img_path = "images/jay.jpg",align = True)
print(result.shape)
show_img([result],["face_man"])

