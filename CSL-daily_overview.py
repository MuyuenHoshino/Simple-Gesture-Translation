#file_path = "/root/autodl-tmp/sentence_label/video_map.txt"
file_path = "./daily_real_balanced_map.txt"
pkl_path = "/root/autodl-tmp/sentence_label/csl2020ct_v2.pkl"
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化一个空列表来存储文件数据
index = []
name = []
length = []
word = []
sentence_length = []
gloss = []
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
    index.append(parts[0])
    name.append(parts[1])
    length.append(int(parts[2]))
    gloss.append(parts[3])
    #char = parts[4]
    word.append(parts[4])
    sentence_length.append(len(parts[5]))
    #postag = parts[6]


print(type(word[1]))
print(word[1].split())
#print(length)
print(word[1])
print("start")
print(max(sentence_length))
print(min(length))
print(max(length))
print(sum(length)/len(length))
#print(word[1])

import pickle

# 打开.pkl文件并读取数据
with open(pkl_path, 'rb') as file:
    data = pickle.load(file)
print(type(data))
#print(data)
keys_list = list(data.keys())
#print(keys_list)
# for key in keys_list:
#     print(key)
# print(data['word_map'])
data_word = data['char_map']

print(len(data_word))
print(data_word[4])


# total_train = []

# for i in range(18401):

#     split = word[i].split()
#     for aword in split:
#         if not aword in total_train:
#             total_train.append(aword)

# print(len(total_train))


# total_val = []
# for i in range(18401,18401+1077):

#     split = word[i].split()
#     for aword in split:
#         if not aword in total_val:
#             total_val.append(aword)

# print(len(total_val))

# total_test = []
# for i in range(18401+1077,18401+1077+1176):

#     split = word[i].split()
#     for aword in split:
#         if not aword in total_test:
#             total_test.append(aword)

# print(len(total_test))


# print("overlap")
# #train and val

# num = 0
# for word in total_val:
#     if word in total_train:
#         num = num + 1
# print(num)

# #train and test

# num = 0
# for word in total_test:
#     if word in total_train:
#         num = num + 1
# print(num)



# bos_train = []
# bos_train_num = []


# for i in range(0,len(length)):

#     split = word[i].split()
#     aword = split[0]
#     if not aword in bos_train:
#         bos_train.append(aword)
#         bos_train_num.append(1)
#     else:
#         bos_train_num[bos_train.index(aword)] = bos_train_num[bos_train.index(aword)] + 1

# print(bos_train)

# numbers = bos_train_num
# elements = bos_train


# letter_number_pairs = list(zip(elements, numbers))
# sorted_letter_number_pairs = sorted(letter_number_pairs, key=lambda x: x[1])

# sorted_letters = [pair[0] for pair in sorted_letter_number_pairs]
# sorted_numbers = [pair[1] for pair in sorted_letter_number_pairs]

# for letter, number in zip(sorted_letters, sorted_numbers):
#     print(f"{letter}: {number}")

# temp = 0
# for num in bos_train_num:
#     temp += num
# print(temp)
# print(len(bos_train_num))


# length_train_num = [0]*len(length)
# total = 0


# for i in range(0,len(length)):

#     split = word[i].split()
#     length_num = len(split)
#     length_train_num[length_num] += 1
#     total += length_num

# part_total = 0
# for i in range(0,5):
#     part_total += length_train_num[i] * i
# print(part_total/total)

# part_total = 0
# for i in range(5,10):
#     part_total += length_train_num[i]* i
# print(part_total/total)

# part_total = 0
# for i in range(10,15):
#     part_total += length_train_num[i]* i
# print(part_total/total)

# part_total = 0
# for i in range(15,20):
#     part_total += length_train_num[i]* i
# print(part_total/total)

# part_total = 0
# for i in range(20,25):
#     part_total += length_train_num[i]* i
# print(part_total/total)

# part_total = 0
# for i in range(25,31):
#     part_total += length_train_num[i]
# print(part_total/total)