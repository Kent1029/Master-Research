import dlib
import cv2
from PIL import Image, ImageChops, ImageEnhance
import os
import glob
import numpy as np
from tqdm import tqdm
import io



# dlib模型路径
DLIB_MODEL_PATH = f'E:\Research\Master-Research\ELA extract\shape_predictor_81_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

def get_dlib_face(img):
    """
    调用dlib获取人脸位置
    :param img: 待截取人脸的图片
    :return: 人脸位置对象face，包括(face.left(), face.top()), (face.right(), face.bottom())
    """
    faces = detector(img, 0)
    if len(faces) == 0:
        return None
    else:
        return faces[0]
    
def conservative_crop(img, face, scale=2.0, new_size=(317, 317)):
    """
    FF++论文中裁剪人脸是先找到人脸框，然后人脸框按比例扩大后裁下更大区域的人脸
    :param img: 待裁剪人脸图片
    :param face: dlib获取的人脸区域
    :param scale: 扩大比例
    :return: 裁剪下来的人脸区域，大小默认为(256,256)，Implementation detail中说预测的mask上采样为256*256，所以截取的人脸应该也是这个大小
    """

    height, width = img.shape[:2]

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    cropped = cv2.resize(img[y1:y1 + size_bb, x1:x1 + size_bb, :], new_size)

    return cropped


def ELA(png_image_path):
        #  載入PNG圖像
        image = Image.open(png_image_path)

        # 将图像保存为JPEG并立即重新打开
        jpeg_image = image.convert('RGB')
        jpeg_image_bytes = io.BytesIO()
        jpeg_image.save(jpeg_image_bytes, format='JPEG')
        jpeg_image_bytes.seek(0)
        jpeg_image = Image.open(jpeg_image_bytes)
        jpeg_image.show()

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

        

def main():
    image=cv2.imread("frame_0.png")
    # 用dlib获取人脸
    face = get_dlib_face(image)
    crop_image=conservative_crop(image,face)
    cv2.imwrite("crop_image.png",crop_image)
    ELA("crop_image.png")
    print("face:::",face)
    cv2.imshow("crop_image",crop_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()