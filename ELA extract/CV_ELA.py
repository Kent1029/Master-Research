import cv2
import numpy as np
import io
import os 
from PIL import Image, ImageChops, ImageEnhance

def ELA_opencv(image):
    # 将图像保存为JPEG并立即重新打开
    _, jpeg_image_bytes = cv2.imencode('.jpg', image)
    jpeg_image = cv2.imdecode(jpeg_image_bytes, cv2.IMREAD_COLOR)

    # 計算ELA圖像
    ela_image = cv2.absdiff(image, jpeg_image)
    max_diff = ela_image.max()
    scale = 255.0 / max_diff
    ela_image = (ela_image * scale).clip(0, 255).astype(np.uint8)

    return ela_image


def ELA(png_image_path,ela_output_path):
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
    ela_image_filename = f'23ela_image.jpg'
    ela_image_path = os.path.join(ela_output_path, ela_image_filename)
    ela_image.save(ela_image_path)
    return ela_image_path

if __name__=="__main__":
    image=cv2.imread("frame_0.png", cv2.IMREAD_COLOR)
    ela_image=ELA_opencv(image)
    ela_image2=ELA("frame_0.png",f'E:\Research\Master-Research\ELA extract')

    cv2.imshow("ELA Result", ela_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# 使用示例
# ela_result = ELA_opencv('path_to_your_image.png')
# cv2.imshow("ELA Result", ela_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()