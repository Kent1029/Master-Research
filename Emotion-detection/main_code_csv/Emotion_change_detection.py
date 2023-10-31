import os
import cv2
import csv
import signal
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Constants and configurations
#FOLDER_PATH ='E:\\Research\\dataset\\FaceForensics++\\original_sequences\\youtube\\c23\\videos'
FORMAT = '.mp4'
#CSV_FILE_PATH = 'YT_emotion_change_counts.csv'
EMOTIONS = ('happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral')

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,choices=['YT','DF','F2F','FS','NT'],default='YT',help='指定dataset')
    args = parser.parse_args()
    return args

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

def create_model():
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
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    return model

def detection(file_path, model,face_haar_cascade):
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    emotion_change = 0
    last_emotion = None
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
            if last_emotion is not None and last_emotion != predicted_emotion:
                emotion_change += 1
            last_emotion = predicted_emotion
            
            emotion_counts[predicted_emotion] += 1

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        #cv2.imshow('Facial emotion analysis', resized_img)
        #cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    return emotion_counts,emotion_change

def write_to_csv(filenames, start_index, model,face_haar_cascade):
    with tqdm(filenames[start_index:], desc='Processing', unit='video', unit_scale=True) as pbar:
        with open(CSV_FILE_PATH, 'a' if start_index else 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if start_index == 0:
                writer.writerow(['Video_name', *EMOTIONS, 'emotion_change'])
            for filename in pbar:
                try:
                    emotion_counts, emotion_changes = detection(os.path.join(FOLDER_PATH, filename), model, face_haar_cascade)
                    print(emotion_counts,"emotion_changes:" ,emotion_changes)
                    writer.writerow([filename, *[emotion_counts[emotion] for emotion in EMOTIONS], emotion_changes])
                    csv_file.flush()
                except KeyboardInterrupt:
                    print("接收到中断信号，程式终止")
                    return

def get_filenames(folder_path, file_format):
    return [filename for filename in os.listdir(folder_path) if filename.endswith(file_format)]

if __name__ == '__main__':
    args = args_func()
    if args.dataset == 'YT':
        FOLDER_PATH ='E:\\Research\\dataset\\FaceForensics++\\original_sequences\\youtube\\c23\\videos'
        CSV_FILE_PATH = 'YT_emotion_change_counts.csv'
    elif args.dataset == 'DF':
        FOLDER_PATH ='E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Deepfakes\\c23\\videos'
        CSV_FILE_PATH = 'DF_emotion_change_counts.csv'
    elif args.dataset == 'F2F':
        FOLDER_PATH ='E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Face2Face\\c23\\videos'
        CSV_FILE_PATH = 'F2F_emotion_change_counts.csv'
    elif args.dataset == 'FS':
        FOLDER_PATH ='E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\FaceSwap\\c23\\videos'
        CSV_FILE_PATH = 'FS_emotion_change_counts.csv'
    elif args.dataset == 'NT':
        FOLDER_PATH ='E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\NeuralTextures\\c23\\videos'
        CSV_FILE_PATH = 'NT_emotion_change_counts.csv'


    model = create_model()
    model.load_weights('model.h5')
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    filenames = get_filenames(FOLDER_PATH, FORMAT)

    start_index = 0
    if os.path.isfile(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH, encoding='big5')
        if not df.empty:
            last_filename = df['Video_name'].values[-1]
            start_index = filenames.index(last_filename) + 1
    
    if start_index == 0:
        print("FOLDER_PATH::",FOLDER_PATH)
        print("現在執行：first_write")
        start_index=0
        write_to_csv(filenames, start_index,model,face_haar_cascade)
    else:
        print("FOLDER_PATH::",FOLDER_PATH)
        print("現在執行：斷點續寫second_write")
        write_to_csv(filenames, start_index,model,face_haar_cascade)
    
