import dlib
import cv2
from PIL import Image, ImageChops, ImageEnhance
import os
import glob
import numpy as np
from tqdm import tqdm
import io
from imutils import face_utils
import pandas as pd



# dlib模型路径
DLIB_MODEL_PATH = f'/home/kent/dataset/ELA_data/shape_predictor_81_face_landmarks.dat'
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
    
def crop_region_face(img, landmarks,crop_output_path,counter,type):
    """
    裁剪出landmarks指定的区域。
    :param img: 图像。
    :param landmarks: 面部landmarks。
    :param index: 需要裁剪的landmarks的索引。
    :return: 裁剪出的区域。
    """
    # 获取指定landmarks的坐标
    points = landmarks
    # 获取最小和最大的x、y值
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    # 将坐标限制在图像的大小内
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    # 裁剪区域
    cropped = img[y1:y2, x1:x2]
    # cropped_path="crop_face.png"
    # cv2.imwrite(cropped_path,cropped)

    crop_image_filename = f'{type}_face_image{counter}.png'
    crop_image_path = os.path.join(crop_output_path, crop_image_filename)
    cv2.imwrite(crop_image_path,cropped)

    return crop_image_path


pre_x1= None
pre_y1= None
pre_x2= None
pre_y2= None
def conservative_crop(img, face, scale=1.2, new_size=(256, 256)):
    global pre_x1,pre_y1,pre_x2,pre_y2
    """
    FF++论文中裁剪人脸是先找到人脸框，然后人脸框按比例扩大后裁下更大区域的人脸
    :param img: 待裁剪人脸图片
    :param face: dlib获取的人脸区域
    :param scale: 扩大比例
    :return: 裁剪下来的人脸区域，大小默认为(256,256)，Implementation detail中说预测的mask上采样为256*256，所以截取的人脸应该也是这个大小
    """

    height, width = img.shape[:2]
    if face is None:
        x1 = pre_x1
        y1 = pre_y1
        x2 = pre_x2
        y2 = pre_y2
    else:
        if face.left() is not None:
            x1 = face.left()
            pre_x1 = x1
        else:
            x1 = pre_x1
            
        if face.top() is not None:
            y1 = face.top()
            pre_y1 = y1
        else:
            y1 = pre_y1
            
        if face.right() is not None:
            x2 = face.right()
            pre_x2 = x2
        else:
            x2 = pre_x2
            
        if face.bottom() is not None:
            y2 = face.bottom()
            pre_y2 = y2
        else:
            y2 = pre_y2
        
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    cropped = cv2.resize(img[y1:y1 + size_bb, x1:x1 + size_bb, :], new_size)

    return cropped


def ELA(png_image_path,ela_output_path,counter,type):
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
    #ela_image.show()
    # 保存ELA圖像
    ela_image_filename = f'{type}_ela_image{counter}.jpg'
    ela_image_path = os.path.join(ela_output_path, ela_image_filename)
    ela_image.save(ela_image_path)
    return ela_image_path



       
def get_region_face_landmark_noise_mask(ela_image_path):

    ela_image = Image.open(ela_image_path)
    # 转化为numpy数组
    ela_array = np.array(ela_image) 

    # 平滑处理
    smoothed = cv2.GaussianBlur(ela_array, (7,7), 0)
    
    # 计算噪声图像
    noise = cv2.subtract(ela_array, smoothed)
    
    # 将彩色噪声图像转化为灰度图像
    gray_noise = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)

    # 使用Otsu的方法自动计算最佳阈值
    _, white_noise = cv2.threshold(gray_noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 进行形态学膨胀操作
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(white_noise, kernel)
    
    # 显示ELA图像和噪声图像
    #cv2.imshow('Noise', dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_region_face_landmark_mask(ela_image_path,mask_output_path,counter,type):
    # 載入ELA圖像
    ela_image = cv2.imread(ela_image_path)

    # 3: 將圖像轉換為HSV並對值分量進行二值化
    hsv = cv2.cvtColor(ela_image, cv2.COLOR_BGR2HSV)
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

    mask_image_filename = f'{type}_mask_image{counter}.png'
    mask_image_path = os.path.join(mask_output_path, mask_image_filename)
    cv2.imwrite(mask_image_path,binary_noise)

    return mask_image_path



prev_landmark = None  # 用來存儲上一張照片的 landmark
def get_landmark(frame):
    global prev_landmark  # 使用全域變數 prev_landmark
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(frame, 1)
    #landmarks = list()  # save the landmark

    if len(faces) > 0:
        landmark = predictor(frame, faces[0])
        landmark = face_utils.shape_to_np(landmark)
        prev_landmark = landmark  # 更新 prev_landmark
        return landmark
    else:
        return prev_landmark



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

def element_wise(crop_image_path,mask_image_path, element_wise_path,counter,type):
    # 開啟 JPEG 和 PNG 圖片
    face_image = cv2.imread(crop_image_path)
    mask_image = cv2.imread(mask_image_path)

    # 確保兩張圖片有相同的尺寸，如果不同，你可以調整它們
    # 注意：如果兩張圖片的尺寸不同，你需要確保它們的尺寸一致才能進行逐元素操作

    # 執行逐元素操作，例如相加
    element_wise_image = cv2.add(face_image, mask_image)


    element_wise_filename = f'{type}_element_wise_image{counter}.png'
    element_wise_image_path = os.path.join(element_wise_path, element_wise_filename)
    cv2.imwrite(element_wise_image_path,element_wise_image)

def main():
    image=cv2.imread("frame_0.png")
    # 用dlib获取人脸
    face = get_dlib_face(image)
    crop_image=conservative_crop(image,face)
    crop_landmark=get_landmark(crop_image)
    cropped_path=crop_region_face(crop_image,crop_landmark)

    ela_image_path=ELA(cropped_path)
    get_region_face_landmark_mask(ela_image_path)
    


def get_files_from_split(split):
    """ "
    Get filenames for real and fake samples

    Parameters
    ----------
    split : pandas.DataFrame
        DataFrame containing filenames
    """
    files_1 = split[0].astype(str).str.cat(split[1].astype(str), sep="_")
    files_2 = split[1].astype(str).str.cat(split[0].astype(str), sep="_")
    files_real = pd.concat([split[0].astype(str), split[1].astype(str)]).to_list()
    files_fake = pd.concat([files_1, files_2]).to_list()
    return files_real, files_fake


if __name__=="__main__":
    main()


# 使用conservative_crop之後，cv2.imwrite("crop_image.png",crop_image)儲存切好的臉在一個floder，
# 之後再從這個floder中讀取image來做ela，
# 做玩ela再來做ELA_mask
# 最終可以得到crop_face, ELA, ELA_mask