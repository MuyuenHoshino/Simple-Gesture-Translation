import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
#import mediapipe_def
#import mediapipe as mp
import numpy
from models import Seq2Seq
import pickle
import time
#mp_holistic = mp.solutions.holistic # Holistic model



"""
Implementation of Chinese Sign Language Dataset(50 signers with 5 times)
"""

class CSL_Daily_Continuous_TwoStream_gloss(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_gloss, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 100#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 100#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['gloss_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")
        print(i)

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open(self.corpus_path, 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []

        i = 0

        # 遍历文件的每一行
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            else:
                i = i + 1
            
            parts = line.strip().split('|')
            sentences.append(parts[3].split())
            sentences_length.append(len(parts[3].split()))

        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        #print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    

    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
        return images



    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx + 0#18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478



        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        #print("sequence.shape: ",sequence.shape)
        #print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改
        tokens = torch.LongTensor(self.corpus[real_index])
        len_label = len(tokens)

        #print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence[:,:256], tokens
        #return images, tokens


class CSL_Daily_Continuous_TwoStream_gloss_simple(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_gloss_simple, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 100#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 100#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['gloss_map']

        self.dict_length = len(self.word_list) + 3

        with open('./daily_gloss_dict.pkl', 'rb') as file:
            self.dict = pickle.load(file)


        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open(self.corpus_path, 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []

        i = 0

        # 遍历文件的每一行
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            else:
                i = i + 1
            
            parts = line.strip().split('|')
            sentences.append(parts[3].split())
            sentences_length.append(len(parts[3].split()))

        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        #print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    

    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            #print("image type:",type(image))

            # 将图像转换为NumPy数组
            image = np.array(image)

            # 将NumPy数组转换为PyTorch张量
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = image.float() / 255.0
            #print("image type:",type(image))


            images.append(image)

        #print(images[0])
        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        #print("数据类型：", images.dtype)
        #print("图像形状：", images.shape)
        return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx + 0#18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478



        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)
        # print(selected_pic_folder)
        #print("sequence.shape: ",sequence.shape)
        #print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改
        tokens = torch.LongTensor(self.corpus[real_index])
        len_label = len(tokens)

        #print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence[:,:256], tokens
        #return images, tokens



class CSL_Daily_Continuous_TwoStream(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 200#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 1176#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open(self.corpus_path, 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []

        i = 0

        # 遍历文件的每一行
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            else:
                i = i + 1
            
            parts = line.strip().split('|')
            sentences.append(parts[5].split())
            sentences_length.append(len(parts[5].split()))

        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    

    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            #print("image type:",type(image))

            # 将图像转换为NumPy数组
            image = np.array(image)

            # 将NumPy数组转换为PyTorch张量
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = image.float() / 255.0
            #print("image type:",type(image))


            images.append(image)

        #print(images[0])
        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        #print("数据类型：", images.dtype)
        print("图像形状：", images.shape)
        return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx #+ 18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478



        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        print("sequence.shape: ",sequence.shape)
        print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改
        tokens = torch.LongTensor(self.corpus[real_index])
        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens


class CSL_Daily_Continuous_TwoStream_balanced(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_balanced, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18603#18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 200#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 200#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open("./daily_balanced_map.txt", 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []
        self.index = []
        i = 0

        # 遍历文件的每一行
        for line in lines:
            i = i + 1
            
            parts = line.strip().split('|')
            #print(parts)
            self.index.append(int(parts[0]))
            sentences.append(parts[5].split())
            sentences_length.append(len(parts[5].split()))
        #print(index[0])
        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    

    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            #print("image type:",type(image))

            # 将图像转换为NumPy数组
            image = np.array(image)

            # 将NumPy数组转换为PyTorch张量
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = image.float() / 255.0
            #print("image type:",type(image))


            images.append(image)

        #print(images[0])
        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
        #print("数据类型：", images.dtype)
        print("图像形状：", images.shape)
        return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx #+ 18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478

        
        tokens = torch.LongTensor(self.corpus[real_index])
        real_index = self.index[real_index]
        print("real index:",real_index)
        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        print("sequence.shape: ",sequence.shape)
        print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改

        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens





class CSL_Daily_Continuous_TwoStream(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 200#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 200#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open(self.corpus_path, 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []

        i = 0

        # 遍历文件的每一行
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            else:
                i = i + 1
            
            parts = line.strip().split('|')
            sentences.append(parts[5].split())
            sentences_length.append(len(parts[5].split()))

        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
        return images

    # def read_images_new(self, folder_path):
    #     images = []
    #     for i in range(0, self.frames):
    #         image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
    #         #print("image type:",type(image))

    #         # 将图像转换为NumPy数组
    #         image = np.array(image)

    #         # 将NumPy数组转换为PyTorch张量
    #         image = torch.from_numpy(image)
    #         image = image.permute(2, 0, 1)
    #         image = image.float() / 255.0
    #         #print("image type:",type(image))


    #         images.append(image)

    #     #print(images[0])
    #     images = torch.stack(images, dim=0)
    #     # switch dimension
    #     images = images.permute(1, 0, 2, 3)
    #     # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
    #     return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx #+ 18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478



        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        print("sequence.shape: ",sequence.shape)
        print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改
        tokens = torch.LongTensor(self.corpus[real_index])
        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens


class CSL_Daily_Continuous_TwoStream_balanced(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_balanced, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18603#18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 200#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 200#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open("./daily_balanced_map.txt", 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []
        self.index = []
        i = 0

        # 遍历文件的每一行
        for line in lines:
            i = i + 1
            
            parts = line.strip().split('|')
            #print(parts)
            self.index.append(int(parts[0]))
            sentences.append(parts[5].split())
            sentences_length.append(len(parts[5].split()))
        #print(index[0])
        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
        return images
    

    # def read_images_new(self, folder_path):
    #     images = []
    #     for i in range(0, self.frames):
    #         image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
    #         #print("image type:",type(image))

    #         # 将图像转换为NumPy数组
    #         image = np.array(image)

    #         # 将NumPy数组转换为PyTorch张量
    #         image = torch.from_numpy(image)
    #         image = image.permute(2, 0, 1)
    #         image = image.float() / 255.0
    #         #print("image type:",type(image))


    #         images.append(image)

    #     #print(images[0])
    #     images = torch.stack(images, dim=0)
    #     # switch dimension
    #     images = images.permute(1, 0, 2, 3)
    #     # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
    #     return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx #+ 18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478

        
        tokens = torch.LongTensor(self.corpus[real_index])
        real_index = self.index[real_index]
        print("real index:",real_index)
        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        print("sequence.shape: ",sequence.shape)
        print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改

        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens





class CSL_Daily_Continuous_TwoStream_256(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=256, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_256, self).__init__()

        self.data_path = "/root/autodl-tmp/picture_new_256"
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = 256

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 200#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 200#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open(self.corpus_path, 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []

        i = 0

        # 遍历文件的每一行
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            else:
                i = i + 1
            
            parts = line.strip().split('|')
            sentences.append(parts[5].split())
            sentences_length.append(len(parts[5].split()))

        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
        return images

    # def read_images_new(self, folder_path):
    #     images = []
    #     for i in range(0, self.frames):
    #         image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
    #         #print("image type:",type(image))

    #         # 将图像转换为NumPy数组
    #         image = np.array(image)

    #         # 将NumPy数组转换为PyTorch张量
    #         image = torch.from_numpy(image)
    #         image = image.permute(2, 0, 1)
    #         image = image.float() / 255.0
    #         #print("image type:",type(image))


    #         images.append(image)

    #     #print(images[0])
    #     images = torch.stack(images, dim=0)
    #     # switch dimension
    #     images = images.permute(1, 0, 2, 3)
    #     # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
    #     return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx #+ 18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478



        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        print("sequence.shape: ",sequence.shape)
        print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改
        tokens = torch.LongTensor(self.corpus[real_index])
        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens



class CSL_Daily_Continuous_TwoStream_256_balanced(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_256_balanced, self).__init__()

        self.data_path = "/root/autodl-tmp/picture_new_256"
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = 256

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18603#18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 200#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 200#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open("./daily_balanced_map.txt", 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []
        self.index = []
        i = 0

        # 遍历文件的每一行
        for line in lines:
            i = i + 1
            
            parts = line.strip().split('|')
            #print(parts)
            self.index.append(int(parts[0]))
            sentences.append(parts[5].split())
            sentences_length.append(len(parts[5].split()))
        #print(index[0])
        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
        return images
    

    # def read_images_new(self, folder_path):
    #     images = []
    #     for i in range(0, self.frames):
    #         image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
    #         #print("image type:",type(image))

    #         # 将图像转换为NumPy数组
    #         image = np.array(image)

    #         # 将NumPy数组转换为PyTorch张量
    #         image = torch.from_numpy(image)
    #         image = image.permute(2, 0, 1)
    #         image = image.float() / 255.0
    #         #print("image type:",type(image))


    #         images.append(image)

    #     #print(images[0])
    #     images = torch.stack(images, dim=0)
    #     # switch dimension
    #     images = images.permute(1, 0, 2, 3)
    #     # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
    #     return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx #+ 18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478

        
        tokens = torch.LongTensor(self.corpus[real_index])
        real_index = self.index[real_index]
        print("real index:",real_index)
        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        print("sequence.shape: ",sequence.shape)
        print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改

        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens






class CSL_Daily_Continuous_TwoStream_real(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_real, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18400
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 100#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 100#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open('dataset_real_train.txt', 'r') as file:
            self.train_list = file.readlines()
        with open('dataset_real_val.txt', 'r') as file:
            self.val_list = file.readlines()
        with open('dataset_real_test.txt', 'r') as file:
            self.test_list = file.readlines()

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open(self.corpus_path, 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []

        i = 0

        # 遍历文件的每一行
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            else:
                i = i + 1
            
            parts = line.strip().split('|')
            sentences.append(parts[5].split())
            sentences_length.append(len(parts[5].split()))

        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
        return images

    # def read_images_new(self, folder_path):
    #     images = []
    #     for i in range(0, self.frames):
    #         image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
    #         #print("image type:",type(image))

    #         # 将图像转换为NumPy数组
    #         image = np.array(image)

    #         # 将NumPy数组转换为PyTorch张量
    #         image = torch.from_numpy(image)
    #         image = image.permute(2, 0, 1)
    #         image = image.float() / 255.0
    #         #print("image type:",type(image))


    #         images.append(image)

    #     #print(images[0])
    #     images = torch.stack(images, dim=0)
    #     # switch dimension
    #     images = images.permute(1, 0, 2, 3)
    #     # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
    #     return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = self.train_list[idx].strip()
        elif self.mode=="val":
            real_index = self.val_list[idx].strip()
        else:
            real_index = self.test_list[idx].strip()



        selected_pic_folder = self.data_path + "/" + real_index
        selected_npy = self.keypoints_path + "/" + real_index + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        print("sequence.shape: ",sequence.shape)
        print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改
        if real_index == "000000":
            real_index_num = 0
        else:
            real_index_num = int(real_index.lstrip("0"))
        tokens = torch.LongTensor(self.corpus[real_index_num])
        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens




class CSL_Daily_Continuous_TwoStream_real_char(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_real_char, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 18400
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open('dataset_real_train.txt', 'r') as file:
            self.train_list = file.readlines()
        with open('dataset_real_val.txt', 'r') as file:
            self.val_list = file.readlines()
        with open('dataset_real_test.txt', 'r') as file:
            self.test_list = file.readlines()

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open(self.corpus_path, 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []

        i = 0

        # 遍历文件的每一行
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            else:
                i = i + 1
            
            parts = line.strip().split('|')
            sentences.append(parts[3].split())
            sentences_length.append(len(parts[5].split()))

        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
        return images

    # def read_images_new(self, folder_path):
    #     images = []
    #     for i in range(0, self.frames):
    #         image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
    #         #print("image type:",type(image))

    #         # 将图像转换为NumPy数组
    #         image = np.array(image)

    #         # 将NumPy数组转换为PyTorch张量
    #         image = torch.from_numpy(image)
    #         image = image.permute(2, 0, 1)
    #         image = image.float() / 255.0
    #         #print("image type:",type(image))


    #         images.append(image)

    #     #print(images[0])
    #     images = torch.stack(images, dim=0)
    #     # switch dimension
    #     images = images.permute(1, 0, 2, 3)
    #     # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
    #     return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = self.train_list[idx].strip()
        elif self.mode=="val":
            real_index = self.val_list[idx].strip()
        else:
            real_index = self.test_list[idx].strip()



        selected_pic_folder = self.data_path + "/" + real_index
        selected_npy = self.keypoints_path + "/" + real_index + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)

        print("sequence.shape: ",sequence.shape)
        print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改
        if real_index == "000000":
            real_index_num = 0
        else:
            real_index_num = int(real_index.lstrip("0"))
        tokens = torch.LongTensor(self.corpus[real_index_num])
        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens





class CSL_Daily_Continuous_TwoStream_real_balanced(Dataset):
    def \
            __init__(self,pkl_path,corpus_path,data_path,keypoints_path, frames=64, mode="train",transform=None):
        super(CSL_Daily_Continuous_TwoStream_real_balanced, self).__init__()

        self.data_path = data_path
        self.pkl_path = pkl_path
        self.keypoints_path = keypoints_path
        self.corpus_path = corpus_path
        self.transform = transform
        self.frames = frames

        self.mode = mode


        #方便整除
        self.total_num = 20000

        # 根据任务不同划分训练集测试集的大小，0.8*50*5=200，训练集每个句子对应200个样本
        if self.mode=="train":
            #self.num = int(self.total_num * 0.7)
            self.num = 20112#18401
        elif mode=="val":
            #self.num = int(self.total_num * 0.2)
            self.num = 200#1077
        elif mode=="test":
            #self.num = int(self.total_num * 0.1)
            self.num = 200#1176
        else:
            print("mode error")
        # dictionary
        print("num",self.num)

        with open(pkl_path, 'rb') as file:
            pkl_data = pickle.load(file)
        
        self.word_list = pkl_data['word_map']

        self.dict_length = len(self.word_list) + 3

        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2} 


        i = 3
        for word in self.word_list:
            self.dict[word] = i
            i = i + 1
        print("dict ready")

        #self.corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"

        with open("./daily_real_balanced_map.txt", 'r') as file:
            lines = file.readlines()

        sentences = []
        sentences_length = []
        self.index = []
        i = 0

        # 遍历文件的每一行
        for line in lines:
            i = i + 1
            
            parts = line.strip().split('|')
            #print(parts)
            self.index.append(int(parts[0]))
            sentences.append(parts[5].split())
            sentences_length.append(len(parts[5].split()))
        #print(index[0])
        max_sentence_length = max(sentences_length) + 2
        print("max sentence length ",max_sentence_length)

        self.corpus = []
        self.unknown = []

        for sentence in sentences:
            tokens = [self.dict['<sos>']]
            for token in sentence:
                if token in self.dict:
                    tokens.append(self.dict[token])
                else:
                    self.unknown.append(token)
            # add eos
            tokens.append(self.dict['<eos>'])
            self.corpus.append(tokens)

        print("length corpus:",len(self.corpus))
        print("test1: ",self.corpus[0])
        #time.sleep(5)
        print("corpus ready")
        #time.sleep(5)
        #print("unknown:",self.unknown)

        unique_unknown_list = []
        for item in self.unknown:
            if item not in unique_unknown_list:
                unique_unknown_list.append(item)

        print(unique_unknown_list)
        print(len(unique_unknown_list))


        for tokens in self.corpus:
            if len(tokens) < max_sentence_length:
                for i in range(len(tokens),max_sentence_length):
                    tokens.insert(i, self.dict['<pad>'] )

        print("test2: ",self.corpus[0])




    def __len__(self):
        # 100*200=20000
        return self.num


    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        # switch dimension
        images = images.permute(1, 0, 2, 3)
        # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
        return images
    

    # def read_images_new(self, folder_path):
    #     images = []
    #     for i in range(0, self.frames):
    #         image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
    #         #print("image type:",type(image))

    #         # 将图像转换为NumPy数组
    #         image = np.array(image)

    #         # 将NumPy数组转换为PyTorch张量
    #         image = torch.from_numpy(image)
    #         image = image.permute(2, 0, 1)
    #         image = image.float() / 255.0
    #         #print("image type:",type(image))


    #         images.append(image)

    #     #print(images[0])
    #     images = torch.stack(images, dim=0)
    #     # switch dimension
    #     images = images.permute(1, 0, 2, 3)
    #     # print(images.shape)
    #     #print("数据类型：", images.dtype)
    #     print("图像形状：", images.shape)
    #     return images


    def __getitem__(self, idx):
        
        real_index = -1

        if self.mode=="train":
            real_index = idx
        elif self.mode=="val":
            #real_index = idx + int(self.total_num * 0.7)
            real_index = idx #+ 18401
        else:
            #real_index = idx + int(self.total_num * 0.9)
            real_index = idx + 19478

        
        tokens = torch.LongTensor(self.corpus[real_index])
        real_index = self.index[real_index]
        print("real index:",real_index)
        selected_pic_folder = self.data_path + "/" + "{:06}".format(real_index)
        selected_npy = self.keypoints_path + "/" + "{:06}".format(real_index) + ".npy"

        
        sequence = numpy.load(selected_npy, allow_pickle=True)
        images = self.read_images_new(selected_pic_folder)


        print("train:",real_index)
        #print("sequence.shape: ",sequence.shape)
        #print("images.shape: ",images.shape)
        # 给定文件夹（索引类别）进行读取，其中250个视频（否）
        ## images = self.read_images(selected_folder)
        # images = self.read_images_new(selected_folder)
        #sequence = self.read_images_new(selected_folder)
        # print(selected_folder, int(idx/self.videos_per_folder))
        # print(self.corpus['{:06d}'.format(int(idx/self.videos_per_folder))])
        #改

        len_label = len(tokens)

        print("length token:",len_label)

        sequence = torch.tensor(sequence, dtype=torch.float32)
        # return images, tokens, len_label, len_voc
        return images, sequence, tokens




# Test
if __name__ == '__main__':
    # Path setting
    data_path = "/root/autodl-tmp/picture_new"
    pkl_path = "/root/autodl-tmp/sentence_label/csl2020ct_v2.pkl"
    keypoints_path = "/root/autodl-tmp/keypoints_new"
    corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"
    transform = transforms.Compose([transforms.Resize([256, 256]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = CSL_Daily_Continuous_TwoStream_real_balanced(data_path=data_path, pkl_path=pkl_path,
        corpus_path=corpus_path, frames=64, mode="train",keypoints_path = keypoints_path,transform=transform)
    #print(len(dataset.corpus))
    # a,b,c = train_set[0]
    # print(a)
    # print(a.shape)
    # print(b)
    # print(b.shape)
    # print(c)
    # print(c.shape)

    # with open('./daily_gloss_dict.pkl', 'wb') as file:
    #     pickle.dump(train_set.dict, file)
    # with open('./daily_gloss_dict.pkl', 'rb') as file:
    #     loaded_dict = pickle.load(file)

    # print(loaded_dict['你们'])
    # #print(len(loaded_dict))
    # dataset = CSL_Continuous_TwoStream(data_path=data_path, dict_path=dict_path,
    #     corpus_path=corpus_path, frames=48, mode="train",pic_path = pic_path,transform = transform)
    
    img,seq,tokens = dataset[30]
    print(img.shape)
    print(tokens)
    print("start------------------")
    #print(img)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Seq2Seq.CSL_Daily_TwoStream_Transformer().to(device)

    # load_path = "/root/autodl-tmp/models/CSL-Daily_TwoStream/CSL-Daily_TwoStream_epoch025.pth"
    # model.load_state_dict(torch.load(load_path)) 

    # model.eval()
    # img = img.unsqueeze(0)
    # seq = seq.unsqueeze(0)
    # tokens_new = tokens[0:2].unsqueeze(0)
    # tokens = tokens.unsqueeze(0)

    
    # img = img.to(device)
    # seq = seq.to(device)
    # tokens = tokens.to(device)
    # tokens_new = tokens_new.to(device)

    # #print("img",img.shape)
    # #print("tokens",tokens.shape)
    # outputs = model(img,seq,tokens)
    # outputs2 = model(img,seq,tokens)
    # print(" ")
    # outputs_new = model(img,seq,tokens_new)

    # print("result-----------")
    # print("old")
    # print(outputs[0][0].shape)
    # print(outputs[0][0])
    # print(outputs2[0])
    # print("new")
    # print(outputs_new[0][0].shape)
    # print(outputs_new[0][0])

    # print("index")
    # max_index = torch.argmax(outputs[1][0])
    # max_index2 = torch.argmax(outputs2[1][0])
    # max_index_new = torch.argmax(outputs_new[1][0])


    # print(max_index,max_index2,max_index_new)

    # print("tokens",tokens)
    # print("tokens_new",tokens_new)
    # i = 0
    # for i in range(9):
    #     max_index = torch.argmax(outputs[i][0])
    #     print(max_index)
    

    # output_dim = outputs.shape[-1]
    # outputs = outputs[1:].view(-1, output_dim)
    # tokens = tokens.permute(1,0)[1:].reshape(-1)

    # print(outputs.shape)
    # print(tokens.shape)
    
    