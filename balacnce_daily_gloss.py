file_path = "/root/autodl-tmp/sentence_label/video_map.txt"
pkl_path = "/root/autodl-tmp/sentence_label/csl2020ct_v2.pkl"
import random
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
    #gloss.append(parts[3])
    #char = parts[4]
    word.append(parts[3])
    sentence_length.append(len(parts[3]))
    #postag = parts[6]


all_by_bos = []
bos_train = []
bos_train_num = []
temp = []
for i in range(0,18401):

    split = word[i].split()
    aword = split[0]
    if not aword in bos_train:
        bos_train.append(aword)
        bos_train_num.append(1)
        temp = [i]
        all_by_bos.append(temp)
    else:
        bos_train_num[bos_train.index(aword)] = bos_train_num[bos_train.index(aword)] + 1
        all_by_bos[bos_train.index(aword)].append(i)

print(all_by_bos[3])




save_lines = []

max = 200
min = 10

all_by_bos_changed = []

for bos_list in all_by_bos:
    if len(bos_list) > max:
        random.shuffle(bos_list)  # 随机打乱列表元素的顺序
        bos_list = bos_list[:max]
    if len(bos_list) < min:
        num_to_copy = min - len(bos_list)
        elements_to_copy = random.choices(bos_list, k=num_to_copy)
        bos_list.extend(elements_to_copy)
    all_by_bos_changed.append(bos_list)

print(all_by_bos_changed[5])

sum = 0

for bos_list in all_by_bos_changed:
    sum += len(bos_list)

print(sum)

with open("daily_balanced_map_by_gloss.txt", 'w') as file:
    for bos_list in all_by_bos_changed:
        for index in bos_list:
            file.write(lines[index+1])



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