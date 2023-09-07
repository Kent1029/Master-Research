import cv2
import dlib
import csv
import os 
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

# 在開始循環之前創建一個CSV文件
csv_file = open('Deepfakes_raw_Eyes_Blink2.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# 指定資料夾path和檔案名稱format
folder_path = 'E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Deepfakes\\raw\\videos'
format = '.mp4'

# 添加CSV文件的列標題
#csv_writer.writerow(['frame','EyeAspectRatio', 'Label'])
csv_writer.writerow(['frame','EyeAspectRatio','left_eye_landmarks', 'right_eye_landmarks', 'Label'])


# 初始化眨眼檢測器
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "shape_predictor_81_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 定義一些參數，例如眨眼閾值和眨眼幀數
eye_aspect_ratio_threshold = 0.2  # 設置眨眼的閾值
consecutive_frames = 2  # 连续帧数以确认眨眼



def get_filenames(folder_path, format):
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(format):
            filenames.append(filename)
    return filenames



def eye_aspect_ratio(eye):
    # 将 dlib 'points' 转换为 tuple 或者 numpy 数组
    eye = np.array([(p.x, p.y) for p in eye])
    # 计算眼睛的垂直距离
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # 计算眼睛的水平距离
    C = distance.euclidean(eye[0], eye[3])

    # 计算眨眼比率
    ear = (A + B) / (2.0 * C)

    return ear

# def extract_eye_landmarks(file_path):
#     video_capture =cv2.VideoCapture(file_path)
#     left_eye_landmarks = []  # 存储左眼特征点坐标
#     right_eye_landmarks = []  # 存储右眼特征点坐标
#     while True:
#         ret, frame = video_capture.read()  # 從視頻流中讀取一幀
#         if not ret:
#             print("extract_eye_landmarks Video stream ended.")
#             break
#         # 将图像转换为灰度图像
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # 使用人脸检测器检测人脸
#         faces = detector(gray)
        
#         # 假设只有一个人脸
#         if len(faces) == 1:
#             face = faces[0]
            
#             # 使用面部关键点检测器检测面部特征点
#             shape = predictor(gray, face)
            
#             # 提取左眼和右眼的特征点坐标（示例中为68点面部关键点检测器）
#             left_eye_landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
#             right_eye_landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
        
#         cv2.imshow("extract_eye_landmarks Image", frame)
#     return left_eye_landmarks, right_eye_landmarks

def detection(file_path):
    consecutive_blinks = 0
    video_capture =cv2.VideoCapture(file_path)
    # 遍歷視頻幀
    while True:
        ret, frame = video_capture.read()  # 從視頻流中讀取一幀
        if not ret:
            print("Video stream ended.")
            break
        #left_eye_landmarks, right_eye_landmarks=extract_eye_landmarks(file_path)
        # 检测人臉
        faces = detector(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_faces = detector(gray)

        # 遍歷每個檢測到的人臉
        for face in faces:
            if len(faces) == 1:
                gray_faces = gray_faces[0]
                
                # 使用面部关键点检测器检测面部特征点
                gray_shape = predictor(gray, gray_faces)
                
                # 提取左眼和右眼的特征点坐标（示例中为68点面部关键点检测器）
                left_eye_landmarks = [(gray_shape.part(i).x, gray_shape.part(i).y) for i in range(36, 42)]
                right_eye_landmarks = [(gray_shape.part(i).x, gray_shape.part(i).y) for i in range(42, 48)]
            
            shape = predictor(frame, face)
            left_eye = [shape.part(i) for i in range(42, 48)]
            right_eye = [shape.part(i) for i in range(36, 42)]

            # 計算左眼和右眼的眨眼比率
            left_eye_ratio = eye_aspect_ratio(left_eye)
            right_eye_ratio = eye_aspect_ratio(right_eye)
            eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

            if eye_avg_ratio < eye_aspect_ratio_threshold:
                consecutive_blinks += 1
                if consecutive_blinks >= consecutive_frames:
                    # 眨眼標籤為真
                    label = 1
            else:
                consecutive_blinks = 0
                label = 0

            # 在此處進行模型訓練或寫入標籤和特徵數據到文件
            csv_writer.writerow([frame,eye_avg_ratio, left_eye_landmarks, right_eye_landmarks,label])
            #csv_writer.writerow([frame,eye_avg_ratio,label])

        # if not ret:
        #     print("Video stream ended.")
        #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("gray_shape Image", gray)
        cv2.imshow("Image", frame)

    #csv_file.close()
    # 釋放視頻流
    video_capture.release()
    cv2.destroyAllWindows()



if __name__=='__main__':
    # 調用function獲取滿足條件的filename
    filenames = get_filenames(folder_path, format)
    for filename in tqdm(filenames, desc='Processing'):#加入tqdm可以有進度條
        print(filename)
        file_path = os.path.join(folder_path, filename)
        detection(file_path)
    csv_file.close()