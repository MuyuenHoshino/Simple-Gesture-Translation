import jieba
from nltk.translate.bleu_score import sentence_bleu
import torch
source = r'猫坐在垫子上'  # source
target = 'the cat is on the mat'  # target
inference = 'the cat is on the mat'  # inference

# 分词
source_fenci = ' '.join(jieba.cut(source))
target_fenci = ' '.join(jieba.cut(target))
inference_fenci = ' '.join(jieba.cut(inference))

# reference是标准答案 是一个列表，可以有多个参考答案，每个参考答案都是分词后使用split()函数拆分的子列表
# # 举个reference例子
# reference = [['this', 'is', 'a', 'duck']]
reference = []  # 给定标准译文
candidate = []  # 神经网络生成的句子
# 计算BLEU
reference.append(target_fenci.split())
candidate = (inference_fenci.split())

print(reference)
print(candidate)
score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
score2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0))
score3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0))
score4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1))
reference.clear()
print('Cumulate 1-gram :%f' \
      % score1)
print('Cumulate 2-gram :%f' \
      % score2)
print('Cumulate 3-gram :%f' \
      % score3)
print('Cumulate 4-gram :%f' \
      % score4)


a = [['班主任', '王', '老师', '仔细', '的', '教导', '，', '很', '亲切', '，', '让', '我', '一直', '记得', '。']]
b = ['班主任', '王', '老师', '仔细', '的', '教导', '很', '亲切', '，', '让', '我', '一直', '记得']
score4 = sentence_bleu(a, b, weights=(0, 0, 0, 1))
print('Cumulate 4-gram :%f' \
      % score4)


import numpy as np

A = torch.tensor([[1],[3]], dtype=torch.float32)

# 创建一个长度为2的张量B
B = torch.tensor([6,7], dtype=torch.float32)
result = torch.cat((A, B.view(2,-1)), dim=1)
print(result)


import multiprocessing
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision.transforms as transforms

criterion =  torch.nn.CTCLoss()
loss = criterion