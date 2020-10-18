#!c:/Users/chenc/.pyvenvs/NinjaMudras/Scripts/python.exe

import globalvar
import os, shutil
label_list = globalvar.label_list
purpose_dict = globalvar.purpose_dict
original_dataset_dir = globalvar.original_dataset_dir
dataset_dir = globalvar.dataset_dir
workspace_dir = globalvar.workspace_dir


batch_size_v = 50
img_x = globalvar.img_x
img_y = globalvar.img_y


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

# 儲存最好的網路模型權重
best_model_h5 = 'ninja-mudras-model-best.h5'
best_model_h5 = os.path.join(model_dir, best_model_h5)
model_mckp = keras.callbacks.ModelCheckpoint(best_model_h5,
                                             monitor='val_categorical_accuracy',
                                             save_best_only=True,
                                             mode='max')



from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

# 問：改成多元分類是否要更改損失函數？
# 答：參考他人專案改成 'categorical_crossentropy'
model.compile(optimizer=optimizers.Adam(),
              loss=losses.CategoricalCrossentropy(),
              metrics=[metrics.CategoricalAccuracy()])

import pandas as pd
import os

# 讀取最佳的網路權重(直接指定檔案，因為剛剛我們只儲存了最佳的模型)
model.load_weights(best_model_h5)

import numpy as np
from tensorflow.keras.preprocessing import image
import cv2


# 選取 0,1號 WebCam
camera_A = cv2.VideoCapture(0)
camera_B = cv2.VideoCapture(1)
# 設定影像的尺寸大小
camera_A.set(cv2.CAP_PROP_FRAME_WIDTH,  160)
camera_A.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
camera_B.set(cv2.CAP_PROP_FRAME_WIDTH,  160)
camera_B.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

player_num = 1

if player_num == 1:
    while globalvar.leave_game != True:
        
        # 從WebCam讀取一張圖片
        suc, frame_A = camera_A.read()
        # 顯示圖片
        cv2.imshow('frame', frame_A)


        cv2.waitKey(100)

        frame_A = cv2.cvtColor(frame_A, cv2.COLOR_BGR2RGB)
        frame_A = cv2.resize(frame_A, (img_x, img_y), interpolation=cv2.INTER_NEAREST)
        frame_A = np.array(frame_A) / 255
        frame_A = image.img_to_array(frame_A)
        frame_A = np.expand_dims(frame_A, axis = 0)

        predict_output_A = model.predict(frame_A)
        max_probablility_index_A = predict_output_A.argmax()

        max_probablility_A = np.max(predict_output_A)
        print("Player A:", label_list[max_probablility_index_A], ":", "{:.2%}".format(max_probablility_A), end="\r")


if player_num == 2:
    while globalvar.leave_game != True:
        
        # 從WebCam讀取一張圖片
        suc, frame_A = camera_A.read()
        suc, frame_B = camera_B.read()
        # 顯示圖片
        cv2.imshow('frame', frame_A)
        cv2.imshow('frame', frame_B)


        cv2.waitKey(100)

        frame_A = cv2.cvtColor(frame_A, cv2.COLOR_BGR2RGB)
        frame_A = cv2.resize(frame_A, (img_x, img_y), interpolation=cv2.INTER_NEAREST)
        frame_A = np.array(frame_A) / 255
        frame_A = image.img_to_array(frame_A)
        frame_A = np.expand_dims(frame_A, axis = 0)

        frame_B = cv2.cvtColor(frame_B, cv2.COLOR_BGR2RGB)
        frame_B = cv2.resize(frame_B, (img_x, img_y), interpolation=cv2.INTER_NEAREST)
        frame_B = np.array(frame_B) / 255
        frame_B = image.img_to_array(frame_B)
        frame_B = np.expand_dims(frame_B, axis = 0)

        predict_output_A = model.predict(frame_A)
        max_probablility_index_A = predict_output_A.argmax()

        predict_output_B = model.predict(frame_B)
        max_probablility_index_B = predict_output_B.argmax()

        max_probablility_A = np.max(predict_output_A)
        max_probablility_B = np.max(predict_output_B)

        print("Player A:", label_list[max_probablility_index_A], ":", "{:.2%}".format(max_probablility_A), "|",
              "Player B:", label_list[max_probablility_index_B], ":", "{:.2%}".format(max_probablility_B), end="\r")

camera_A.release()
camera_B.release()
cv2.destroyAllWindows()
