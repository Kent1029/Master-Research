import os
import shutil
from tqdm import tqdm

# 輸入目錄
real_input_dir = '/home/kent/Baseline_method/CADDM/train_images/Celeb-DF/all_Celeb-real/'
fake_input_dir = '/home/kent/Baseline_method/CADDM/train_images/Celeb-DF/all_Celeb-synthesis/'

# 輸出目錄
real_train_dir = f'/home/kent/Baseline_method/CADDM/train_images/Celeb-DF/Celeb-real/'
real_test_dir = f'/home/kent/Baseline_method/CADDM/test_images/Celeb-DF/Celeb-real/'
fake_train_dir = f'/home/kent/Baseline_method/CADDM/train_images/Celeb-DF/Celeb-synthesis/'
fake_test_dir = f'/home/kent/Baseline_method/CADDM/test_images/Celeb-DF/Celeb-synthesis/'

# 創建輸出目錄
os.makedirs(real_train_dir, exist_ok=True)
os.makedirs(real_test_dir, exist_ok=True)
os.makedirs(fake_train_dir, exist_ok=True)
os.makedirs(fake_test_dir, exist_ok=True)


def split_folders(input_dir, train_dir, test_dir):
    # 獲取所有子目錄
    folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # 計算訓練集和測試集的大小
    num_folders = len(folders)
    num_train = int(num_folders * 0.8)
    
    # 分割子目錄
    train_folders = folders[:num_train]
    test_folders = folders[num_train:]
    
    # 複製子目錄到輸出目錄
    for d in tqdm(train_folders):
        shutil.copytree(os.path.join(input_dir, d), os.path.join(train_dir, d))
        
    for d in tqdm(test_folders):
        shutil.copytree(os.path.join(input_dir, d), os.path.join(test_dir, d))

split_folders(real_input_dir, real_train_dir, real_test_dir)
split_folders(fake_input_dir, fake_train_dir, fake_test_dir)

print('Files copied successfully!')
