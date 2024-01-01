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

data_path = "/root/autodl-tmp/picture"
save_path = "/root/autodl-tmp//keypoints_720-720"
frames = 48
data_num = 100


listdir_root = os.listdir(data_path)


for i in range(0,100):
    file_name = '{:06d}'.format(i)
    sentence_path = data_path + "/" + file_name
    sentence_path_save = save_path + "/" + file_name
    if not os.path.exists(sentence_path_save):
        os.mkdir(sentence_path_save)
    listdir_sentence = os.listdir(sentence_path)
    for j in range(250):
        folder_path = sentence_path + "/" + listdir_sentence[j]
        folder_path_save = sentence_path_save + "/" + listdir_sentence[j]
        #print(folder_path_save + ".npy")
        if os.path.exists(folder_path_save + ".npy"):
            print("yes")
        else:
            sequence = []
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for k in range(48):
                    pic_name = '{:06d}'.format(k)
                    pic_path = folder_path + "/" + pic_name + ".jpg"
                    print(pic_path)
                    img = Image.open(pic_path)
                    transform = transforms.CenterCrop(720)
                    img = transform(img)
                    frame = cap = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                    image, results = mediapipe_def.mediapipe_detection(frame, holistic)
                    keypoints = mediapipe_def.extract_keypoints(results)
                    sequence.append(keypoints)
            sequence = np.array(sequence)
            np.save(folder_path_save,sequence)


