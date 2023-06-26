import pandas as pd
from statsmodels.multivariate.manova import MANOVA

# 讀取資料
DF_data = pd.read_csv('DF_emotion_counts.csv')
YT_data = pd.read_csv('YT_emotion_counts.csv')

# 提取YT dataset和DF dataset的表情資料
YT_emotions = YT_data[['happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral']]
DF_emotions = DF_data[['happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral']]

# 將資料格式轉換為多變量分析所需的形式
data_dict = {'YT': YT_emotions, 'DF': DF_emotions}
df = pd.concat(data_dict)

# 加入dataset欄位
df['dataset'] = ['YT'] * len(YT_emotions) + ['DF'] * len(DF_emotions)

# 進行多變量方差分析（MANOVA）
maov = MANOVA.from_formula('happy + angry + sad + surprise + disgust + fear + neutral ~ dataset', data=df)
results = maov.mv_test()

# 印出結果
print(results)