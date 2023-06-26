import pandas as pd
from scipy.stats import f_oneway

# 读取数据
data = pd.read_csv('YT_emotion_counts.csv')

# 提取感情类别的列
emotion_columns = ['happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral']
emotion_data = data[emotion_columns]
print("len(data[emotion_columns]",len(data[emotion_columns]))

# 执行方差分析
f_statistic, p_value = f_oneway(*[emotion_data[column] for column in emotion_columns])

#all dataset mean value
dataset_mean = emotion_data.mean().mean()
print("Dataset overall mean:", dataset_mean)


# 计算平均值
mean_values = emotion_data.mean()
print("Mean values:")
print(mean_values)

# 计算最大值
max_values = emotion_data.max()
print("Max values:")
print(max_values)


# 计算整体最大值
dataset_max = emotion_data.max().max()

print("Dataset overall maximum:", dataset_max)


print("F-statistic:", f_statistic)
print("p-value:", p_value)

std_values = emotion_data.std()
print("Standard deviation values:")
print(std_values)

# 合并所有情绪类别的数据
all_data = pd.concat([emotion_data[column] for column in emotion_columns], axis=0)

# 计算整体标准差
dataset_std = all_data.std()

print("Dataset overall standard deviation:", dataset_std)