import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import time

# load model
model = load_model("best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path ="000_003.mp4"
fps = 20  # 設定取樣率為25

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, fps)

start_time = time.time()
fframe_count = 0
while True:
    
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    
    if not ret:
        break
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels, verbose=0)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        # print(predicted_emotion)

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    fframe_count += 1

    if cv2.waitKey(int(1000 / 30)) == ord('q'):  # wait until 'q' key is pressed
        break
 

print("fframe_count",fframe_count)

end_time = time.time()
elapsed_time = end_time - start_time
ffps = fframe_count / elapsed_time
print("平均預測幀率:", ffps)



fps = cap.get(cv2.CAP_PROP_FPS) 
print("幀率:", fps)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = frame_count / fps
print("影片時間（秒）:", video_duration)

cap.release()
cv2.destroyAllWindows