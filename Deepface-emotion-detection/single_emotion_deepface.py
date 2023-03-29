import cv2
from deepface import DeepFace
import numpy as np

img = cv2.imread('images/trump.png')
face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")   # 載入人臉模型
faces = face_cascade.detectMultiScale(img)    # 偵測人臉

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)    # 利用 for 迴圈，抓取每個人臉屬性，繪製方框
try:
    emotion = DeepFace.analyze(img, actions=['emotion'])  # 情緒
    age = DeepFace.analyze(img, actions=['age'])          # 年齡
    race = DeepFace.analyze(img, actions=['race'])        # 人種
    gender = DeepFace.analyze(img, actions=['gender'])    # 性別

    print(emotion[0]['dominant_emotion'])
    print(age[0]['age'])
    print(race[0]['dominant_race'])
    print(gender[0]['gender'])
except:
    pass

cv2.imshow('Emotion Detect', img)
cv2.waitKey(0)
cv2.destroyAllWindows()