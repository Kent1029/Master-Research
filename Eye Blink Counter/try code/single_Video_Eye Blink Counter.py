import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time

video_name=input("請輸入影片檔名：")
video_name=video_name+".mp4"
cap = cv2.VideoCapture(video_name)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=False)

idList = [22, 23, 24, 26, 110, 130, 157, 158, 159, 160, 161, 243, 249, 263, 362, 373, 374,380, 381, 382, 384, 385, 386, 387, 388, 390, 466, 467]
#print("SORT",sorted(set(idList)))
ratioList = []
blinkCounter = 0
counter = 0
color = (0,200,0)
#start_time = time.time() #可以計算while-loop 執行秒數
while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        #cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 讓影片重複撥放
        #end_time = time.time() #可以計算while-loop 執行秒數
        #total_time = end_time - start_time #可以計算while-loop 執行秒數
        #print('程式執行時間：', total_time, '秒') #可以計算while-loop 執行秒數
        print("眨眼次數：",blinkCounter)
        break

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False) #改為False可以消除臉上的468檢測點

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5,color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtVer, _ = detector.findDistance(leftUp, leftDown)
        lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35 and counter == 0:
            blinkCounter += 1
            color = (0, 0, 255)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                color = (0,200,0)

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100),
                           colorR=color)

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, imgPlot], 1, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 1, 1)

    cv2.imshow("Image", imgStack)
    cv2.waitKey(25)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
duration = total_frames / fps
print('影片秒數：', duration)