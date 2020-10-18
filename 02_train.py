#!c:/Users/chenc/.pyvenvs/NinjaMudras/Scripts/python.exe

import globalvar
import os, shutil
label_list = globalvar.label_list
purpose_dict = globalvar.purpose_dict
original_dataset_dir = globalvar.original_dataset_dir
dataset_dir = globalvar.dataset_dir
workspace_dir = globalvar.workspace_dir

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# 
# # 忍術結印手勢判斷
# 
# %% [markdown]

# %% [markdown]


# %% [markdown]
# 
# ### 使用 ImageDataGenerator 產生器從dataset資料夾中讀取影像
# 

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size_v = 100
img_x = globalvar.img_x
img_y = globalvar.img_y

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,         # 隨機水平翻轉
    rotation_range=5,            # 隨機旋轉 ±10度
    zoom_range=0.1,               # 隨機縮放 ±15%
    width_shift_range=.15,        # 隨機水平移動 ±15%
    height_shift_range=.15,       # 隨機垂直移動 ±15%
) #設定訓練、測試資料的 Python 產生器，並將圖片像素值依 1/255 比例重新壓縮到 [0, 1]
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(
    purpose_dict["train"],              # 目標目錄
    target_size=(img_x, img_y),  # 調整所有影像大小成 150x150
    batch_size=batch_size_v,
    shuffle=True,        # 順序洗牌
    class_mode='categorical',
    # color_mode='grayscale'
    color_mode='rgb'
)    # 因為使用二元交叉熵 binary_crossentropy 作為損失值，所以需要二位元標籤


validation_generator = validation_datagen.flow_from_directory(
    purpose_dict["validation"],
    target_size=(img_x, img_y),
    # target_size=(256, 124),  # 調整所有影像大小成 256x124(4:3)
    batch_size=batch_size_v,
    shuffle=True,        # 順序洗牌
    class_mode='categorical',
    # color_mode='grayscale'
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    purpose_dict["test"],
    target_size=(img_x, img_y),
    # target_size=(256, 124),  # 調整所有影像大小成 256x124(4:3)
    batch_size=batch_size_v,
    shuffle=True,        # 順序洗牌
    class_mode='categorical',
    # color_mode='grayscale'
    color_mode='rgb'
)

# %% [markdown]
# 
# ## <kbd>Step 2</kbd> 神經網路 建立與設定
# 
# %% [markdown]
# 
# ### 繪製一個小型的卷積神經網路(CNN)
# 

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# statement expand 描述式展開
# keras.models.Sequential().add(...)
#                               keras.layers.Conv2D(...)

inputs = keras.Input(shape=(img_x, img_y, 3))
x = layers.Conv2D( 64, (5,5), activation='relu')(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(32, (5,5), activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(128, (5,5), activation='relu')(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D( 64, (5,5), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(len(label_list),       activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(label_list), activation='softmax')(x) # 輸出神經元個數採用我們的label數量

model = keras.Model(inputs, outputs, name='model')
model.summary()  # 查看模型摘要

# %% [markdown]
# 
# ### 建立model儲存資料夾
# 

# %%
model_dir = os.path.join(workspace_dir, 'train-logs')

# model_dir 不存在時，建立新資料夾
if not (os.path.isdir(model_dir)): 
    os.mkdir(model_dir)

# %% [markdown]
# 
# ### 建立回調函數(Callback function)
# 

# %%
# 將訓練紀錄儲存為TensorBoard的紀錄檔
log_dir = os.path.join(model_dir, 'model')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)

# 只儲存最好的網路模型權重
best_model    = 'ninja-mudras-model'
best_model    = os.path.join(model_dir, best_model)
best_model_h5 = 'ninja-mudras-model-best.h5'
best_model_h5 = os.path.join(model_dir, best_model_h5)
model_mckp = keras.callbacks.ModelCheckpoint(best_model_h5,
                                             monitor='val_categorical_accuracy',
                                             save_best_only=True,
                                             mode='max')



# %% [markdown]
# 
# ### 設定model的 損失函數、優化器
# 

# %%
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

# 問：改成多元分類是否要更改損失函數？
# 答：參考他人專案改成 'categorical_crossentropy'
model.compile(optimizer=optimizers.Adam(),
              loss=losses.CategoricalCrossentropy(),
              metrics=[metrics.CategoricalAccuracy()])

# %% [markdown]
# ### 看看產生器的輸出結果

# %%
# 這邊需要PIL module
# 用pip install Pillow安裝
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# %% [markdown]
# ### 調整 model 以使用批次量產生器 (開始訓練)

# %%
history = model.fit(
                    train_generator,
                    epochs=1000,
                    validation_data=validation_generator,
                    callbacks=[model_cbk,model_mckp]
                    )

# 儲存 model
model.save(best_model)

# %% [markdown]
# ### 顯示訓練結果

# %%
import pandas as pd
import os

# 讀取最佳的網路權重(直接指定檔案，因為剛剛我們只儲存了最佳的模型)
model.load_weights(best_model_h5)

# 驗證在測試集資料上
loss, acc = model.evaluate(test_generator)



dict_result = {"Accuracy":acc, "Loss":loss}
print("\n\n\n")
print(
    "Predict result:",
    dict_result
    )
# os.system("pause")


#region     [顯示訓練和驗證週期的損失值和準確度曲線]

# import matplotlib.pyplot as plt

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()

#endregion  [顯示訓練和驗證週期的損失值和準確度曲線]



