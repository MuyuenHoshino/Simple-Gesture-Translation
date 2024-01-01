import multiprocessing
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.transforms as transforms
import mediapipe as mp
import mediapipe_def
mp_holistic = mp.solutions.holistic # Holistic model

data_path = "/root/autodl-tmp/picture_new"
save_path = "/root/autodl-tmp//keypoints_new"
frames = 64
data_num = 20654


listdir_root = os.listdir(data_path)


for i in range(0,20654):
    file_name = '{:06d}'.format(i)
    sentence_path = data_path + "/" + file_name
    # sentence_path_save = save_path + "/" + file_name
    # if not os.path.exists(sentence_path_save):
    #     os.mkdir(sentence_path_save)
    if os.path.exists(save_path + "/" + file_name + ".npy"):
        continue
    sequence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for k in range(64):
                pic_name = '{:06d}'.format(k)
                pic_path = sentence_path + "/" + pic_name + ".jpg"
                print(pic_path)
                img = Image.open(pic_path)
                #transform = transforms.CenterCrop(720)
                #img = transform(img)
                frame = cap = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                image, results = mediapipe_def.mediapipe_detection(frame, holistic)
                keypoints = mediapipe_def.extract_keypoints(results)
                sequence.append(keypoints)
        sequence = np.array(sequence)
        np.save(save_path + "/" + file_name + ".npy",sequence)


