import multiprocessing
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.transforms as transforms

file_path = "/root/autodl-tmp/sentence_label/video_map.txt"
pkl_path = "/root/autodl-tmp/sentence_label/csl2020ct_v2.pkl"
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化一个空列表来存储文件数据
#index = []
name = []
length = []
word = []

i = 0

# 遍历文件的每一行
for line in lines:

    print(i)

    if i == 0:
        i = i + 1
        continue
    else:
        i = i + 1
    
    # 使用竖线分割每一行的数据
    parts = line.strip().split('|')
    
    # 提取每个字段的值
    #index.append(parts[0])
    name.append(parts[1])
    length.append(int(parts[2]))
    #gloss = parts[3]
    #char = parts[4]
    word.append(parts[5])

data_path = "/root/autodl-tmp/newdata/frames_512x512"
save_path = "/root/autodl-tmp/picture_new_256"
frames = 256
sample_size = 256


import shutil


for index in range(len(os.listdir(data_path))):

    if index%100 == 0:
        print(index)

    #print('{:06d}'.format(index))
    video_path = data_path + "/" + name[index]
    #print(video_path)
   
    
    
    
    lenB = length[index]
    pic_list = os.listdir(video_path)
    #print("111",pic_list)
    pic_list = sorted(pic_list, key=lambda x: int(x.split('.')[0]))
    #print("222",pic_list)

    if lenB >= frames:
        for o in range(0, int(lenB - frames)):

            del pic_list[np.random.randint(0, len(pic_list))]
    else:
        for o in range(0,int(0 - lenB + frames)):
            pic_list.append("./white_image.jpg")

    #print(pic_list)

    #print(len(pic_list))

    os.makedirs(save_path + "/" + "{:06d}".format(index), exist_ok=True) 
    
    for i in range(len(pic_list)):
        if i >= lenB:
            file_to_copy = pic_list[i]
        else:
            file_to_copy = data_path + "/" + name[index] + "/" + pic_list[i]
        #print(file_to_copy)
        folder_path = save_path + "/" + "{:06d}".format(index) + "/" + "{:06d}".format(i) + ".jpg"
        shutil.copy(file_to_copy, folder_path)


