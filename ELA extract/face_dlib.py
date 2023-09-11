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
        #jpeg_image.show()

        # 計算ELA圖像
        ela_image = ImageChops.difference(image, jpeg_image)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        ela_image.show()

        # 保存ELA圖像
        ela_image_path = "0_ela.jpg"
        ela_image.save(ela_image_path)

        print("ELA圖像已保存到", ela_image_path)

        
def get_mask(img):
    _, binary_b = cv2.threshold(img[:, :, 0], 0, 255, cv2.THRESH_BINARY)
    _, binary_g = cv2.threshold(img[:, :, 1], 0, 255, cv2.THRESH_BINARY)
    _, binary_r = cv2.threshold(img[:, :, 2], 0, 255, cv2.THRESH_BINARY)
    mask = np.clip(binary_b + binary_g + binary_r, 0, 255)

    res = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        # 是否为凸包
        ret = cv2.isContourConvex(contours[c])
        # 凸包检测
        points = cv2.convexHull(contours[c])

        # 返回的points形状为(凸包边界点个数，1,2)
        # 使用fillPoly函数应该把前两个维度对调
        points = np.transpose(points, (1, 0, 2))
        # print(points.shape)
        # 描点然后用指定颜色填充点围成的图形内部，生成原始的mask
        cv2.fillPoly(res, points, color=(255, 255, 255))

    return np.expand_dims(res, axis=2)


def mask(ela_image):
    
    ret, binary_noise = cv2.threshold(ela_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("binary_noise",binary_noise)

    # 4: Binary Noise = MorphClose(Binary Noise, Cross)
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_CLOSE, cross_kernel)

    # 5: Binary Noise = MorphClose(Binary Noise, Square)
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_CLOSE, square_kernel)

    # 6: Binary Noise = MorphOpen(Binary Noise)
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_OPEN, None)

    # 7: Binary Noise = MorphClose(Binary Noise, Ellipse)
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_CLOSE, ellipse_kernel)

    # 8: Binary Noise = MorphErode(Binary Noise)
    binary_noise = cv2.erode(binary_noise, None)

    # 9: Binary Noise = MorphDilate(Binary Noise)
    binary_noise = cv2.dilate(binary_noise, None)

    # 10: Binary Noise = GaussianBlur(Binary Noise)
    binary_noise = cv2.GaussianBlur(binary_noise, (3, 3), 0)

    # 11: M = Normalize(Binary Noise, alpha=0, beta=1)
    normalized = cv2.normalize(binary_noise, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalized

def main():
    image=cv2.imread("frame_0.png")
    # 用dlib获取人脸
    face = get_dlib_face(image)
    crop_image=conservative_crop(image,face)
    #cv2.imshow("crop_image",crop_image)
    cv2.imwrite("crop_image.png",crop_image)
    ELA("crop_image.png")
    ela_image=cv2.imread("0_ela.jpg",0)
    normalized=mask(ela_image)
    cv2.imshow('Normalized', normalized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 保存mask
    
if __name__=="__main__":
    main()


# 使用conservative_crop之後，cv2.imwrite("crop_image.png",crop_image)儲存切好的臉在一個floder，
# 之後再從這個floder中讀取image來做ela，
# 做玩ela再來做ELA_mask
# 最終可以得到crop_face, ELA, ELA_mask