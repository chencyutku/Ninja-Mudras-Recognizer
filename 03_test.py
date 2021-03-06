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


from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    purpose_dict["test"],
    target_size=(img_x, img_y),
    # target_size=(256, 124),  # 調整所有影像大小成 256x124(4:3)
    batch_size=batch_size_v,
    shuffle=True,        # 順序洗牌
    class_mode='categorical',
    # color_mode='grayscale',
    color_mode='rgb')




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
x = layers.Flatten()(x)
x = layers.Dense(len(label_list),        activation='relu')(x)
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

# 驗證在測試集資料上
loss, acc = model.evaluate(test_generator)


dict_result = {"Accuracy":acc, "Loss":loss}
print("\n\n\n")
print(
    "Predict result:",
    dict_result
    )
