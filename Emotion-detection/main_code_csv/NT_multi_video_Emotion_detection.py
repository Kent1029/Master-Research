'''
加入了signal配合try expect:處理Ctrl+C終止訊號
感謝https://github.com/atulapra/Emotion-detection.git
https://github.com/atulapra/Emotion-detection/tree/master

python file_name.py

'''


import os
import cv2
import csv
import time
import signal
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import  load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

# 定義一個訊號處理的函數
def signal_handler(signal, frame):
    print("程式終止")
    # 在這裡執行釋放資源的清理工作
    # 關閉文件、釋放資源
    # 退出程式
    csv_file.close()
    sys.exit(0)

# 創建一個訊號處理的函數
signal.signal(signal.SIGINT, signal_handler)


def detection(file_path):
    # 创建一个空的字典来保存情绪计数
    emotion_counts = {'happy': 0, 'angry': 0, 'sad': 0, 'surprise': 0, 'disgust': 0, 'fear': 0, 'neutral': 0}
    cap = cv2.VideoCapture(file_path)
    while True:
        ret, test_img = cap.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = roi_gray.reshape((1, 48, 48, 1))
            img_pixels = img_pixels.astype('float32')
            img_pixels /= 255.0

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotions = ('happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral')
            predicted_emotion = emotions[max_index]
            emotion_counts[predicted_emotion] += 1

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        #cv2.imshow('Facial emotion analysis', resized_img)
        #cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    return emotion_counts


def get_filenames(folder_path, format):
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(format):
            filenames.append(filename)
    return filenames


def first_write():
    pbar = tqdm(filenames, desc='Processing', unit='video', unit_scale=True)
    try:
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Video_name', 'happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral'])
            for filename in pbar:
                try:
                    print(filename)
                    file_path = os.path.join(folder_path, filename)
                    emotion_counts=detection(file_path)
                    
                    print("情緒次數統計：",emotion_counts)
                    print("csv_file_path:",csv_file_path)
                    # 將變數寫入 CSV 文件
                    writer.writerow([filename, emotion_counts['happy'], emotion_counts['angry'], emotion_counts['sad'],
                                    emotion_counts['surprise'], emotion_counts['disgust'], emotion_counts['fear'], emotion_counts['neutral']])
                    csv_file.flush() #刷新文件的緩衝區，將資料存入csv 
                except KeyboardInterrupt:
                    #捕獲終止的訊號
                    print("接收到中断信号，程式终止")
                    pbar.close()  # 手動關閉tqdm的bar
                    csv_file.close()  # 關閉文件
                    sys.exit(0)
    except KeyboardInterrupt:
        print("接收到中断信号，程式终止")
        sys.exit(0)


def second_write():
    pbar = tqdm(filenames[start_index:], desc='Processing', unit='video', unit_scale=True)
    try:
        with open(csv_file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for filename in pbar:
                try:
                    print(filename)
                    file_path = os.path.join(folder_path, filename)
                    emotion_counts=detection(file_path)
                    
                    print("情緒次數統計：",emotion_counts)
                    # 將變數寫入 CSV 文件
                    writer.writerow([filename, emotion_counts['happy'], emotion_counts['angry'], emotion_counts['sad'],
                                    emotion_counts['surprise'], emotion_counts['disgust'], emotion_counts['fear'], emotion_counts['neutral']])
                    csv_file.flush() #刷新文件的緩衝區，將資料存入csv 
                except KeyboardInterrupt:
                    #捕獲終止的訊號
                    print("接收到中断信号，程式终止")
                    pbar.close()  # 手動關閉tqdm的bar
                    csv_file.close()  # 關閉文件
                    sys.exit(0)
    except KeyboardInterrupt:
        print("接收到中断信号，程式终止")
        sys.exit(0)

# main

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
print("model",model)
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
# load model
#model = load_model("emotion_model.h5")
model.load_weights('model.h5')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 指定資料夾path和檔案名稱format
folder_path = 'E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\NeuralTextures\\c23\\videos'


format = '.mp4'

# 調用function獲取滿足條件的filename
filenames = get_filenames(folder_path, format)

csv_file_path = 'NT_emotion_counts.csv'

# 檢查 CSV 文件是否存在
if os.path.isfile(csv_file_path):
    # 讀取 CSV 文件
    df = pd.read_csv(csv_file_path, encoding='big5')
    # 檢查 CSV 文件是否有資料
    if not df.empty:
        # 獲取最後一筆vide_name
        last_filename = df['Video_name'].values[-1]

        # 找到最后一笔数据在 filenames 列表中的索引位置
        start_index = filenames.index(last_filename) + 1

    else:
        start_index = 0
else:
    start_index = 0

# 判断是执行 first_write() 還是 second_write()
if start_index == 0:
    print("現在執行：first_write")
    first_write()
else:
    print("現在執行：斷點續寫second_write")
    second_write()
