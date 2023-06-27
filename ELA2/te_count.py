import pandas as pd
import csv


with open('ELA_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # 將讀取到的資料轉換為資料框
    ELA_data = pd.DataFrame(reader)


ela=ELA_data.values
print("ela:::",ela)
print("ELA_data:::",type(ELA_data))


