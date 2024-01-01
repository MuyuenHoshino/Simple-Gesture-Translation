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

data_path = "/root/autodl-tmp/ISLR"
save_path = "/root/autodl-tmp/ISL_keypoints"
frames = 16



listdir_root = os.listdir(data_path)

for i in range(89,90):
    print(i)
    selected_folder = data_path + "/" + '{:03d}'.format(i)
    #print(selected_folder)
    listdir_selected = os.listdir(selected_folder)
    #print(listdir_selected)
    npy_save_path = save_path+ "/" + '{:03d}'.format(i)
    #print(npy_save_path)
    if not os.path.exists(npy_save_path):
        print("create the npy save folder")
        os.mkdir(npy_save_path)
    for j in range(250):
        folder_path = selected_folder + "/" + listdir_selected[j]
        #print(folder_path)
        npy_save_name = npy_save_path + "/" + listdir_selected[j] + ".npy"
        #print(folder_path_save + ".npy")
        if os.path.exists(npy_save_path + "/" + ".npy"):
            print(npy_save_path + "/" + ".npy" + "  already exists")
        else:
            sequence = []
            start = 1
            step = int(len(os.listdir(folder_path)) / frames)
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for i in range(frames):
                    img_path = (folder_path + "/" + '{:06d}.jpg').format(start + i * step)
                    img = Image.open(img_path)
                    frame = cap = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                    image, results = mediapipe_def.mediapipe_detection(frame, holistic)
                    keypoints = mediapipe_def.extract_keypoints(results)
                    sequence.append(keypoints)
            sequence = np.array(sequence)
            np.save(npy_save_name,sequence)

