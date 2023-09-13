import cv2
import os


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



def fake_element_wise(crop_image_path,mask_image_path, element_wise_path,counter):
    # 開啟 JPEG 和 PNG 圖片
    face_image = cv2.imread(crop_image_path)
    mask_image = cv2.imread(mask_image_path)

    # 確保兩張圖片有相同的尺寸，如果不同，你可以調整它們
    # 注意：如果兩張圖片的尺寸不同，你需要確保它們的尺寸一致才能進行逐元素操作

    # 執行逐元素操作，例如相加
    element_wise_image = cv2.add(face_image, mask_image)


    element_wise_filename = f'fake_element_wise_image{counter}.png'
    element_wise_image_path = os.path.join(element_wise_path, element_wise_filename)
    cv2.imwrite(element_wise_image_path,element_wise_image)