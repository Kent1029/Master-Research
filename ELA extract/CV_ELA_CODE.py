import cv2
import numpy as np
from imutils import face_utils
'''
用在CADDM的face_dlib
把counter與type拿掉
直接輸入image得到ELA_element_WISE的image

'''



def ELA(image):
    #ELA use openCV
    # 将图像保存为JPEG并立即重新打开
    _, jpeg_image_bytes = cv2.imencode('.jpg', image)
    jpeg_image = cv2.imdecode(jpeg_image_bytes, cv2.IMREAD_COLOR)

    # 計算ELA圖像
    ela_image = cv2.absdiff(image, jpeg_image)
    max_diff = ela_image.max()
    scale = 255.0 / max_diff
    ela_image = (ela_image * scale).clip(0, 255).astype(np.uint8)

    return ela_image

def get_region_face_landmark_mask(ELA_image):

    # 3: 將圖像轉換為HSV並對值分量進行二值化
    hsv = cv2.cvtColor(ELA_image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    _, binary_noise = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4, 5, 6, 7, 8, 9: 進行形態學操作
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_CLOSE, kernel_cross)
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_CLOSE, kernel_square)
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_OPEN, kernel_square)
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_CLOSE, kernel_ellipse)
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_ERODE, kernel_square)
    binary_noise = cv2.morphologyEx(binary_noise, cv2.MORPH_DILATE, kernel_square)

    # 10: 高斯模糊
    binary_noise = cv2.GaussianBlur(binary_noise, (5, 5), 0)

    # 11: 歸一化
    binary_noise = cv2.normalize(binary_noise, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    binary_noise = (binary_noise * 255).astype(np.uint8)

    return binary_noise


def get_full_face_landmark_mask(srcRgb,landmark):
    mask = np.zeros(srcRgb.shape, dtype=np.uint8)

    points = cv2.convexHull(
        np.array(landmark).astype('float32')
    )
    corners = np.expand_dims(points, axis=0).astype(np.int32)
    cv2.fillPoly(mask, corners, (255,)*3)
    # gaussianblur.
    blured = cv2.GaussianBlur(mask, (5, 5), 3).astype('float32')
    threshold_value = 128  # 設定閾值
    ret, binary_image = cv2.threshold(blured, threshold_value, 255, cv2.THRESH_BINARY)

    #cv2.imshow("blured", binary_image)
    return  binary_image

def element_wise(ELA_image,ELA_make_image):
    # 開啟 JPEG 和 PNG 圖片
    face_image = ELA_image
    mask_image = ELA_make_image

    # 確保兩張圖片有相同的尺寸，如果不同，你可以調整它們
    # 注意：如果兩張圖片的尺寸不同，你需要確保它們的尺寸一致才能進行逐元素操作
    # 擴展 mask_image 的維度
    mask_image_expanded = cv2.merge([mask_image, mask_image, mask_image])
    # 使用 cv2.add() 進行相加操作
    element_wise_image = cv2.add(face_image, mask_image_expanded)

    # 另一種element wise 逐元素的相乘方法
    # result = cv2.multiply(face_image / 255.0, mask_image_expanded / 255.0)
    # element_wise_image = (result * 255).astype(np.uint8)
    return element_wise_image


def ELA_WISE(image): 
    ELA_image=ELA(image)
    ELA_make_image=get_region_face_landmark_mask(ELA_image)
    element_wise_image=element_wise(ELA_image,ELA_make_image)
    return element_wise_image

if __name__=="__main__":
    ELA_WISE()

