import multiprocessing
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.transforms as transforms


data_path = "/root/autodl-tmp/sentences_SLR_dataset/color"
save_path = "/root/autodl-tmp/picture"
frames = 48
sample_size = 128
data_num = 100

transform = transforms.Compose([transforms.Resize([sample_size, sample_size]), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])


class Preprocess:

    def __init__(self, data_num, data_path, save_path, frames=48):
        self.data_num = data_num
        self.data_path = data_path
        self.save_path = save_path
        self.frames = frames


        for i in range(0, self.data_num):
            file_name = '{:06d}'.format(i)
            if not os.path.exists(os.path.join(self.save_path, file_name)):
                os.mkdir(os.path.join(self.save_path, file_name))
            listdir = os.listdir(os.path.join(self.data_path, file_name))
            for j in range(0, len(listdir)):
                if not os.path.exists(os.path.join(self.save_path, file_name, listdir[j][:-4])):
                    os.mkdir(os.path.join(self.save_path, file_name, listdir[j][:-4]))

    def cut_images(self, folder_path, file_name):

        if len(os.listdir(os.path.join(self.save_path, file_name, os.path.basename(folder_path)[:-4]))) == self.frames:
            return

        images = []  # list
        capture = cv2.VideoCapture(folder_path)

        fps_all = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        # 取整数部分
        timeF = int(fps_all / self.frames)
        n = 1

        # 对一个视频文件进行操作
        while capture.isOpened():
            ret, frame = capture.read()
            if ret is False:
                break
            # 每隔timeF帧进行存储操作
            if (n % timeF == 0):
                image = frame  # frame是PIL
                images.append(image)
            n = n + 1

        capture.release()
        lenB = len(images)
        # 将列表随机去除一部分元素，剩下的顺序不变

        for o in range(0, int(lenB - self.frames)):
            # 删除一个长度内随机索引对应的元素，不包括len(images)即不会超出索引
            del images[np.random.randint(0, len(images))]
            # images.pop(np.random.randint(0, len(images)))
        lenF = len(images)

        for i in range(0, lenF):
            basename = os.path.basename(folder_path)[:-4]
            cv2.imwrite(os.path.join(os.path.join(self.save_path, file_name, basename, "{:06}.jpg".format(i))), images[i])
            print(os.path.join(os.path.join(self.save_path, file_name, basename, "{:06}.jpg".format(i))))

    def begin(self, left, right):
        # print(left, right)
        for i in range(left, right):
            file_name = '{:06d}'.format(i)
            listdir = os.listdir(os.path.join(self.data_path, file_name))
            for j in range(0, len(listdir)):
                self.cut_images(os.path.join(self.data_path, file_name, listdir[j]), file_name)


if __name__ == "__main__":

    preprocess = Preprocess(data_num, data_path, save_path, frames)
    n_cpu = multiprocessing.cpu_count()
    queue = []

    if n_cpu > data_num:
        for i in range(0, data_num):
            proc = multiprocessing.Process(target=preprocess.begin, args=(i, i+1))
            queue.append(proc)
    else:
        total = 0
        step = []
        for i in range(0, n_cpu):
            step.append(0)
        for i in range(0, data_num // n_cpu + 1):
            for j in range(0, n_cpu):
                step[j] += 1
                total += 1
                if total == data_num:
                    break
            if total == data_num:
                break

        total = 0
        for i in range(0, n_cpu):
            proc = multiprocessing.Process(target=preprocess.begin, args=(total, total + step[i]))
            print(total, total + step[i])
            queue.append(proc)
            total += step[i]
        print(step)

    for i in range(0, len(queue)):
        queue[i].start()

    for i in range(0, len(queue)):
        queue[i].join()
