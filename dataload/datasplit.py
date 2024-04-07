# coding:utf-8

import os
import numpy as np
import random
from tqdm import tqdm

#3个数据子集的划分比例
train_percent = 0.7
val_percent = 0.15
test_percent = 0.15

rootpath = 'data/airfoil'
dataset_name = 'supercritical_airfoil'

#创建文件
if not os.path.exists(rootpath):
    os.mkdir(rootpath)
file_train = open(os.path.join(rootpath,f'{dataset_name}_train.txt'),mode='w')
file_val = open(os.path.join(rootpath,f'{dataset_name}_val.txt'),mode='w')
file_test = open(os.path.join(rootpath,f'{dataset_name}_test.txt'),mode='w')
 
path_alldata = os.path.join(rootpath,dataset_name)

file_images_real = np.empty([0, 2])
# 文件名，xxxxx.txt
alldata = os.listdir(path_alldata)

#计算各个训练集的长度
len_all = len(alldata)
len_train = len_all*train_percent
len_train = int(len_train)
len_test = len_all*test_percent
len_test = int(len_test)
len_val = len_all*val_percent
len_val = int(len_val)
 
print("len_all = ", len_all)
print("len_train = ", len_train)
print("len_val = ", len_val)
print("len_test = ", len_test)


 
#开始分配数据
train_counts = random.sample(range(0,len_all),len_train)
test_counts = random.sample(range(0,len_all),len_test)
val_counts = random.sample(range(0,len_all),len_val)

# 创建tqdm对象，设置总迭代次数为数据总数
progress_bar = tqdm(total=len(train_counts) + len(test_counts) + len(val_counts), desc='Writing Data', unit='file')

#写入数据
for train_count in train_counts:
    filename = os.path.splitext(alldata[train_count])
    file_train.writelines(f'{filename[0]}\n')
    progress_bar.update(1)
 
for test_count in test_counts:
    # file_test.writelines(f'{file_images_real[test_count][0]}\n')
    filename = os.path.splitext(alldata[test_count])
    file_test.writelines(f'{filename[0]}\n')
    progress_bar.update(1)
 
for val_count in val_counts:
    # file_val.writelines(f'{file_images_real[val_count][0]}\n')
    filename = os.path.splitext(alldata[val_count])
    file_val.writelines(f'{filename[0]}\n')
    progress_bar.update(1)
 
 
progress_bar.close()  # 关闭进度条

file_train.close()
file_test.close()
file_val.close()