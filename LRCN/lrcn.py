import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet34
from tensorflow.keras.layers import Dense, GlobalAveragePooling3D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 資料集路徑
train_dir = 'path_to_train_directory'
test_dir = 'path_to_test_directory'

# 影片參數設定
frames = 16
height = 224
width = 224
channels = 3

# 分類數量
num_classes = 2

# 資料增強
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# 載入訓練集和測試集
train_data = train_datagen.flow_from_directory(train_dir,
                                              target_size=(height, width),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)

test_data = test_datagen.flow_from_directory(test_dir,
                                            target_size=(height, width),
                                            batch_size=32,
                                            class_mode='categorical',
                                            shuffle=False)

# 建立預訓練的 ResNet-34 模型
base_model = ResNet34(weights='imagenet', include_top=False, input_shape=(height, width, channels))

# 凍結預訓練模型的權重
for layer in base_model.layers:
    layer.trainable = False

# 在預訓練模型的輸出層之後添加自定義的分類層
x = base_model.output
x = GlobalAveragePooling3D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 建立整合預訓練模型和自定義分類層的新模型
model = Model(inputs=base_model.input, outputs=predictions)

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(train_data, epochs=10)

# 評估模型
test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# 使用模型進行預測
predictions = model.predict(test_data)
