import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Reshape

# 影片參數設定
frames = 16
height = 224
width = 224
channels = 3

# LSTM 參數設定
lstm_units = 256
dropout_rate = 0.5

# 分類數量
num_classes = 2

# 資料集路徑
train_dir = '/home/kent/dataset/FaceForensics++/train/'
test_dir = '/home/kent/dataset/FaceForensics++/test/'

# 載入訓練集和測試集
def load_data(directory):
    videos = []
    labels = []
    class_labels = os.listdir(directory)
    for i, class_label in enumerate(class_labels):
        class_dir = os.path.join(directory, class_label)
        if os.path.isdir(class_dir):  # 確認路徑是資料夾
            video_files = os.listdir(class_dir)
            for video_file in video_files:
                video_path = os.path.join(class_dir, video_file)
                if video_file.endswith('.mp4'):  # 只處理 .mp4 檔案
                    video = []
                    cap = cv2.VideoCapture(video_path)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, (width, height))
                        frame = frame.astype('float32') / 255.0
                        video.append(frame)
                    cap.release()
                    if len(video) >= frames:
                        video = video[:frames]
                        videos.append(video)
                        labels.append(i)
    videos = np.array(videos)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=num_classes)
    print("Number of videos:", len(videos))
    print("Number of labels:", len(labels))
    return videos, labels


# 載入訓練資料
x_train, y_train = load_data(os.path.join(train_dir, 'real'))
x_fake_train, y_fake_train = load_data(os.path.join(train_dir, 'fake'))
x_train = np.concatenate((x_train, x_fake_train), axis=0)
y_train = np.concatenate((y_train, y_fake_train), axis=0)

# 載入測試資料
x_test, y_test = load_data(os.path.join(test_dir, 'real'))
x_fake_test, y_fake_test = load_data(os.path.join(test_dir, 'fake'))
x_test = np.concatenate((x_test, x_fake_test), axis=0)
y_test = np.concatenate((y_test, y_fake_test), axis=0)

# 建立預訓練的 ResNet-50 模型
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, channels))

# 建立 LRCN 模型
model = Sequential()
model.add(Conv3D(64, (3, 3, 3), activation='relu', input_shape=(frames, height, width, channels)))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(Conv3D(128, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(256, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Conv3D(256, (3, 3, 3), activation='relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
# 在Flatten層後添加Reshape層
model.add(Reshape((frames, -1)))
model.add(LSTM(lstm_units, dropout=dropout_rate))
model.add(Dense(num_classes, activation='softmax'))

# 設定凍結部分權重
for layer in model.layers[:7]:
    layer.trainable = False

# 編譯模型
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, batch_size=32,epochs=10)
model.save('video_lrcn.h5')

# 評估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# 使用模型進行預測
predictions = model.predict(x_test)
