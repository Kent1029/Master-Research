import pandas as pd
import csv


with open('img_array.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # 將讀取到的資料轉換為資料框
    data = pd.DataFrame(reader)

# 取得欄位名稱
columns_data = data.iloc[0].tolist()[:1000]

# 將欄位名稱寫入 "ELA_data.csv" 檔案
with open('ELA_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns_data)

print("columns_data:", columns_data)
print("len_columns_data:", len(columns_data))


