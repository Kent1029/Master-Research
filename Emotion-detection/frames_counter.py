# import os
# import cv2

# folder_path = 'E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Deepfakes\\c40\\videos'

# for filename in os.listdir(folder_path):
#     if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):
#         video_path = os.path.join(folder_path, filename)
#         video = cv2.VideoCapture(video_path)
#         fps = video.get(cv2.CAP_PROP_FPS)
#         print(f"影片 {filename} 的FPS為: {fps}")

#         video.release()

import os
import cv2

#fake_folder_path = 'E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\Face2Face\\c40\\videos'
fake_folder_path = 'E:\\Research\\dataset\\FaceForensics++\\manipulated_sequences\\NeuralTextures\\c40\\videos'
real_folder_path = 'E:\\Research\\dataset\\FaceForensics++\\original_sequences\\youtube\\c40\\videos'
total_fps = 0
video_count = 0

for filename in os.listdir(real_folder_path):
    if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):
        video_path = os.path.join(real_folder_path, filename)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        # print(f"影片 {filename} 的FPS為: {fps}")
        
        total_fps += fps
        video_count += 1

        video.release()

if video_count > 0:
    average_fps = total_fps / video_count
    print(f"\n平均FPS為: {average_fps}")
else:
    print("資料夾中沒有影片檔案。")
