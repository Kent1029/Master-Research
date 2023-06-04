import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time
import os
import csv
from tqdm import tqdm

cap = None
def detection(file_path):
    global cap
    blink_timestamps = []
    cap = cv2.VideoCapture(file_path)
    detector = FaceMeshDetector(maxFaces=1)
    plotY = LivePlot(640, 360, [20, 50], invert=False)

    idList = [22, 23, 24, 26, 110, 130, 157, 158, 159, 160, 161, 243, 249, 263, 362, 373, 374,380, 381, 382, 384, 385, 386, 387, 388, 390, 466, 467]
    #print("SORT",sorted(set(idList)))
    ratioList = []
    blinkCounter = 0
    counter = 0
    color = (0,200,0)
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            #cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 讓影片重複撥放
            #print("眨眼次數：",blinkCounter)
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
                blink_timestamps.append(time.time())#計算平均眨眼秒數

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
    
    time_diffs = [blink_timestamps[i] - blink_timestamps[i-1] for i in range(1, len(blink_timestamps))]#計算平均眨眼秒數
    avg_time_diff = sum(time_diffs) / len(time_diffs) if len(time_diffs) > 0 else 0 #計算平均眨眼秒數
    return blinkCounter ,avg_time_diff

# 計算影片秒數
def second(file_path):
    global cap
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    #print('影片秒數：', duration)
    return duration

def get_filenames(folder_path, format):
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(format):
            filenames.append(filename)
    return filenames

# 指定資料夾path和檔案名稱format
folder_path = 'E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Deepfakes\\c40\\videos'
format = '.mp4'

# 調用function獲取滿足條件的filename
filenames = get_filenames(folder_path, format)

csv_file_path = 'Deepfakes_c40_Eyes_Blink_Counter.csv'

# 逐一輸出filename
with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # 寫入欄位標題
    writer.writerow(['檔案名稱','影片秒數', '眨眼次數', '平均眨眼秒數'])
    for filename in tqdm(filenames, desc='Processing'):#加入tqdm可以有進度條
        print(filename)
        file_path = os.path.join(folder_path, filename)
        blinkCounter,avg_time_diff=detection(file_path)
        seconds=second(file_path)
        
        print("眨眼次數：",blinkCounter)
        print("平均眨眼秒數：",avg_time_diff)
        print('影片秒數：', seconds)
        # 將變數寫入 CSV 文件
        writer.writerow([filename,seconds, blinkCounter, avg_time_diff])
