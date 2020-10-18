#!c:/Users/chenc/.pyvenvs/NinjaMudras/Scripts/python.exe

import globalvar
import os, shutil
label_list = globalvar.label_list
purpose_dict = globalvar.purpose_dict
original_dataset_dir = globalvar.original_dataset_dir
dataset_dir = globalvar.dataset_dir
workspace_dir = globalvar.workspace_dir

# 
# ## <kbd>Step 1</kbd> 用自己的資料來源建立dataset
# 
# %% [markdown]
# 
# ### 分配並創建存放資料的dataset工作區資料夾
# 

# %%











# %% [markdown]
# 
# ### 複製圖片到訓練、驗證和測試集資料夾
# 

# %%
#region     [複製圖片到 訓練、驗證、測試 資料夾]

import random

index_base   = 0
index_max    = 400
index_offset_dict = {
    "train":      (index_max // 2),
    "validation": (index_max // 4),
    "test":       (index_max // 4)
}
total_index = sum(index_offset_dict.values())

for label in label_list:

    fnames = ['{}_{:05d}.jpg'.format(label,i) for i in range(total_index)]

    # 把編號洗牌過再複製
    random.shuffle(fnames)

    for purpose in purpose_dict:
        for fname in fnames[0:index_offset_dict[purpose]]:
            src = os.path.join(original_dataset_dir, label, fname)
            dst = os.path.join(dataset_dir, purpose, label, fname)
            if(os.path.exists(src)):
                shutil.copyfile(src, dst)
        del fnames[0:index_offset_dict[purpose]]


#endregion  [複製圖片到 訓練、驗證、測試 資料夾]

# %% [markdown]
# 
# ### 計算每個訓練/驗證/測試分組中的圖片數量，做為資料完整性的檢查：
# 

# %%
#region     [資料完整性檢查]

for purpose in purpose_dict:
    for label in label_list:
        print("{}用的{}照片張數{:3d}".format(purpose,
                                             label,
                                             len(os.listdir(os.path.join(dataset_dir,
                                                                         purpose,
                                                                         label)))))

#endregion  [資料完整性檢查]
