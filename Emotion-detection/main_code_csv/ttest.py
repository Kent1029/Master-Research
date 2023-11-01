import numpy as np

# 定義情緒類別和數量
emotions = ['happy', 'angry', 'sad', 'surprise', 'disgust', 'fear', 'neutral']
n_emotions = len(emotions)

# 初始化 7x7 的矩陣
transition_matrix = np.zeros((n_emotions, n_emotions))

# # 對每支影片進行遍歷
# for movie in movies:
#     for i in range(len(movie) - 1):
#         # 找到起始情緒和目標情緒的索引
#         start_emotion_idx = emotions.index(movie[i])
#         end_emotion_idx = emotions.index(movie[i + 1])
        
#         # 在相應的矩陣元素上加一
#         transition_matrix[start_emotion_idx, end_emotion_idx] += 1

# # 計算轉換機率
# row_sums = transition_matrix.sum(axis=1, keepdims=True)
# transition_prob_matrix = transition_matrix / row_sums

print("轉換次數矩陣：")
print(emotions)
print(transition_matrix)
# print("轉換機率矩陣：")
# print(transition_prob_matrix)
