import cv2
import numpy as np
from imutils import face_utils
import random
import os
import dlib
import albumentations as alb
'''
用在CADDM的face_dlib
把counter與type拿掉
直接輸入image得到ELA_element_WISE的image

'''
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
    
def conservative_crop(img, face, scale=1.2, new_size=(224, 224)):
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

# dlib模型路径
DLIB_MODEL_PATH = f'E:\Research\Master-Research\ELA extract\shape_predictor_81_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

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

  
def crop_region_face(img, landmarks):
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

    #crop_image_filename = f'{type}_face_image{counter}.png'
    #crop_image_path = os.path.join(crop_output_path, crop_image_filename)
    #cv2.imwrite(crop_image_path,cropped)

    return cropped


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


def get_blend_mask(mask):
	H,W=mask.shape
	size_h=np.random.randint(192,257)
	size_w=np.random.randint(192,257)
	mask=cv2.resize(mask,(size_w,size_h))
	kernel_1=random.randrange(5,26,2)
	kernel_1=(kernel_1,kernel_1)
	kernel_2=random.randrange(5,26,2)
	kernel_2=(kernel_2,kernel_2)
	
	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured<1]=0
	
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured,(W,H))
	return mask_blured.reshape((mask_blured.shape+(1,)))

def dynamic_blend(source,target,mask):
	mask_blured = get_blend_mask(mask)
	#mask_blured = np.repeat(mask_blured[:, :, np.newaxis], 3, axis=2)
	blend_list=[0.5]
	#blend_list=[0.25,0.5,0.75,1,1,1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_blured*=blend_ratio
	print("mask:",mask_blured.shape)
	print("source:",source.shape)
	print("target:",target.shape)
	img_blended=(mask_blured * source + (1 - mask_blured) * target)
	return img_blended,mask_blured

class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
	def apply(self,img,**params):
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

		return img_ds

def get_source_transforms():
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

def ELA_WISE(image): 
    face = get_dlib_face(image)
    crop_image=conservative_crop(image,face)
    crop_landmark=get_landmark(crop_image)
    crop_image=crop_region_face(crop_image,crop_landmark) 

    ELA_image=ELA(crop_image)
    ELA_make_image=get_region_face_landmark_mask(ELA_image)
    element_wise_image=element_wise(crop_image,ELA_make_image)
    #target_transforms_image=get_source_transforms(image=crop_image.astype(np.uint8))['image']
    # 使用轉換
    transform = get_source_transforms()
    augmented = transform(image=crop_image)
    target_transforms_image = augmented['image']
    # print("target_transforms_image:", type(target_transforms_image))
    # print("target_transforms_image:", target_transforms_image.shape)
    # #print("target_transforms_image:",type(target_transforms_image))
    # print("crop_image::",type(crop_image))
    # print("crop_image::",crop_image.shape)
    blend_img,mask_blured=dynamic_blend(element_wise_image,target_transforms_image,ELA_make_image)
    blend_img = blend_img.astype(np.uint8)
    mask_blured = mask_blured.astype(np.uint8)
    # 顯示圖片
    #cv2.imshow('ELA_image', ELA_image)
    cv2.imshow('ELA_make_image', ELA_make_image)
    cv2.imshow('element_wise_image', element_wise_image)
    cv2.imshow('target_transforms_image', target_transforms_image)
    cv2.imshow('blend_img', blend_img)
    cv2.imshow('mask_blured', mask_blured)

    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return blend_img

if __name__=="__main__":
    # 讀取圖檔
    img = cv2.imread('frame_0.png')
    blend_img=ELA_WISE(img)
    # 寫入圖檔
    cv2.imwrite('blend_img1.jpg', blend_img)
    # 顯示圖片
    # 按下任意鍵則關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()

