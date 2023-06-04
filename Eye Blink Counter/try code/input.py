import os
#import time

def get_filenames(folder_path, format):
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(format):
            filenames.append(filename)
    return filenames

# 指定資料夾path和檔案名稱format
folder_path = 'E:\\Research\\dataset\\FaceForensics++\\original_sequences\\youtube\\c40\\videos'
format = '.mp4'

# 調用function獲取滿足條件的filename
filenames = get_filenames(folder_path, format)

# 逐一輸出filename
for filename in filenames:
    print(filename)
    #time.sleep(1)
    
