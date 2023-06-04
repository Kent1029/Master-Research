import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 讀取真實影片的CSV檔案
real_data = np.loadtxt('avgblink_youtube_c40_real_data.csv', delimiter=',', dtype=float, skiprows=1)
real_features = real_data[:, 0].reshape(-1, 1)  # 真實影片的眨眼次數特徵
real_labels = np.ones(len(real_data))  # 真實影片的標籤設置為1

# 讀取Deepfake影片的CSV檔案
deepfake_data = np.loadtxt('avgblink_Deepfakes_c40_deepfake_data.csv', delimiter=',', dtype=float, skiprows=1)
deepfake_features = deepfake_data[:, 0].reshape(-1, 1)  # Deepfake影片的眨眼次數特徵
deepfake_labels = np.zeros(len(deepfake_data))  # Deepfake影片的標籤設置為0

# 合併特徵和標籤
features = np.concatenate((real_features, deepfake_features), axis=0)
labels = np.concatenate((real_labels, deepfake_labels), axis=0)

# 將數據分為訓練集和測試集
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

# 創建MLP模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(train_features, train_labels, epochs=2000, batch_size=32, verbose=1)

# 在測試集上評估模型
test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=0)
print('Test accuracy:', test_acc)

# 預測測試集
predictions = model.predict(test_features)

# 計算AUC
auc = roc_auc_score(test_labels, predictions)
print('AUC:', auc)
