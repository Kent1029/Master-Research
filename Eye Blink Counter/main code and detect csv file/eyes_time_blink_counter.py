import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time
import os
import csv
from tqdm import tqdm
import pandas as pd
import argparse


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,choices=['YT','DF','F2F','FS','NT','test'],default='YT',help='指定dataset')
    args = parser.parse_args()
    return args

def detection(file_path, cap):
    blink_timestamps = []
    detector = FaceMeshDetector(maxFaces=1)
    plotY = LivePlot(640, 360, [20, 50], invert=False)
    idList = [22, 23, 24, 26, 110, 130, 157, 158, 159, 160, 161, 243, 249, 263, 362, 373, 374,380, 381, 382, 384, 385, 386, 387, 388, 390, 466, 467]
    ratioList = []
    blinkCounter = 0
    counter = 0
    color = (0,200,0)
    blink_timestamps=[]
    imgPlot=[]
    #The variable for 5 sec countr
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_for_5sec = 5 * fps
    frame_counter = 0
    blink_counter_5sec = 0
    blinks_every_5sec = []

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
                blink_counter_5sec += 1
                color = (0, 0, 255)
                counter = 1
                blink_timestamps.append(time.time())#計算平均眨眼秒數

            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (0,200,0)
            
            frame_counter += 1
            if frame_counter == frames_for_5sec:
                blinks_every_5sec.append(blink_counter_5sec)
                blink_counter_5sec = 0
                frame_counter = 0

            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100),
                            colorR=color)

            imgPlot = plotY.update(ratioAvg, color)
            img = cv2.resize(img, (640, 360))
            imgStack = img
            #imgStack = cvzone.stackImages([img, img], 1, 1)
        else:
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img, img], 1, 1)

        #cv2.imshow("Image", imgStack)
        cv2.waitKey(25)
    
    time_diffs = [blink_timestamps[i] - blink_timestamps[i-1] for i in range(1, len(blink_timestamps))]#計算平均眨眼秒數
    avg_time_diff = sum(time_diffs) / len(time_diffs) if len(time_diffs) > 0 else 0 #計算平均眨眼秒數
    if len(blinks_every_5sec)>0:
        blinks_per_minute = sum(blinks_every_5sec) / (len(blinks_every_5sec))
    elif len(blinks_every_5sec)==0:
        blinks_per_minute = 0
    cap.release()
    cv2.destroyAllWindows()
    try:
        value_0 = blinks_every_5sec[0]
    except IndexError:
        value_0 = 0

    try:
        value_1 = blinks_every_5sec[1]
    except IndexError:
        value_1 = 0

    try:
        value_2 = blinks_every_5sec[2]
    except IndexError:
        value_2 = 0
    return blinkCounter, avg_time_diff,value_0, value_1, value_2,blinks_per_minute

cap = None
def get_video_duration(file_path):
    global cap
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    cap.release()
    cv2.destroyAllWindows()
    return duration

def get_filenames(folder_path, format):
    return [filename for filename in os.listdir(folder_path) if filename.endswith(format)]

def write_to_csv(filenames, folder_path, csv_file_path, start_index):
    mode = 'w' if start_index == 0 else 'a'
    with open(csv_file_path, mode, newline='') as csv_file:
        writer = csv.writer(csv_file)
        if mode == 'w':
            writer.writerow(['Video_ID','Video_Time', 'Blink_count', 'Ave_count',"1st(5 sec)","2st(5 sec)","3st(5 sec)","Avg Time varibale(5 sec)"])##
        
        for filename in tqdm(filenames[start_index:], desc='Processing'):
            file_path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(file_path)
            blinkCounter, avg_time_diff,sec1,sec2,sec3,blinks_per_minute = detection(file_path, cap)
            seconds = get_video_duration(file_path)
            cap.release()
            print('Video_Time：', seconds,"Avg_count：", avg_time_diff,"Blink_count：", blinkCounter,"Avg Time varibale(5sec)",blinks_per_minute)
            writer.writerow([filename, seconds, blinkCounter, avg_time_diff,sec1,sec2,sec3,blinks_per_minute])##

def main(args,folder_paths,csv_file_paths):
    folder_path = folder_paths
    format = '.mp4'
    filenames = get_filenames(folder_path, format)
    csv_file_path = csv_file_paths 
    
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path, encoding='big5')
        if not df.empty:
            last_filename = df['Video_ID'].values[-1]
            start_index = filenames.index(last_filename) + 1
        else:
            start_index = 0
    else:
        start_index = 0

    if start_index == 0:
        print("folder_path::",folder_path)
        print("現在執行first_write")
        start_index=0
        write_to_csv(filenames, folder_path, csv_file_path, start_index)
    else:
        print("folder_path::",folder_path)
        print("現在執行：斷點續寫second_write")
        write_to_csv(filenames, folder_path, csv_file_path, start_index)


if __name__ == '__main__':
    args = args_func()
    if args.dataset == 'YT':
        folder_paths ='E:\\Research\\dataset\\FaceForensics++\\original_sequences\\youtube\\c23\\videos'
        csv_file_paths = 'csv_file/YT_time_Eyes_Blink_Counter.csv'
    elif args.dataset == 'DF':
        folder_paths ='E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Deepfakes\\c23\\videos'
        csv_file_paths = 'csv_file/DF_time_Eyes_Blink_Counter.csv'
    elif args.dataset == 'F2F':
        folder_paths ='E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Face2Face\\c23\\videos'
        csv_file_paths = 'csv_file/F2F_time_Eyes_Blink_Counter.csv'
    elif args.dataset == 'FS':
        folder_paths ='E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\FaceSwap\\c23\\videos'
        csv_file_paths = 'csv_file/FS_time_Eyes_Blink_Counter.csv'
    elif args.dataset == 'NT':
        folder_paths ='E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\NeuralTextures\\c23\\videos'
        csv_file_paths = 'csv_file/NT_time_Eyes_Blink_Counter.csv'
    elif args.dataset == 'test':
        folder_paths ='E:\\Research\\Master-Research\\Emotion-detection\\main_code_csv\\test_video'
        csv_file_paths = 'csv_file/test_time_Eyes_Blink_Counter.csv'


    if not os.path.exists(f"csv_file"):
        os.makedirs(f"csv_file")

    main(args,folder_paths,csv_file_paths)
