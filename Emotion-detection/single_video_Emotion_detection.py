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
model = load_model("emotion_model.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path ="000_003.mp4"

cap = cv2.VideoCapture(video_path)


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

        emotions = ('happy','angry','sad','surprise', 'disgust', 'fear','neutral')
        predicted_emotion = emotions[max_index]
        # print(predicted_emotion)

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    cv2.waitKey(25)

    # if cv2.waitKey(0) == ord('q'):  # wait until 'q' key is pressed
    #     break

cap.release()
cv2.destroyAllWindows