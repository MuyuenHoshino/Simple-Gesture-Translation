

file_path = "/root/autodl-tmp/sentence_label/video_map.txt"
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
    word.append(parts[5])
    sentence_length.append(len(parts[5]))
    #postag = parts[6]

dict = {}

for i in range(len(index)):
    dict[name[i]] = index[i]

#print(dict)



file_path = "/root/autodl-tmp/sentence_label/split_1.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化一个空列表来存储文件数据
name = []
att = []
train_list = []
val_list = []
test_list = []

strange_list = []

i=0
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
    
    name.append(parts[0])
    att.append(parts[1])

for i in range(len(name)):

    if name[i] in dict:
        if att[i] == "train":
            train_list.append(dict[name[i]])
        if att[i] == "dev":
            val_list.append(dict[name[i]])
        if att[i] == "test":
            test_list.append(dict[name[i]])
    else:
        strange_list.append(name[i])
        
print(len(train_list),len(val_list),len(test_list))
print(strange_list)

with open('dataset_real_train.txt', 'w') as file:
    for item in train_list:
        file.write(str(item) + '\n')

with open('dataset_real_test.txt', 'w') as file:
    for item in test_list:
        file.write(str(item) + '\n')

with open('dataset_real_val.txt', 'w') as file:
    for item in val_list:
        file.write(str(item) + '\n')
    
with open('dataset_real_train.txt', 'r') as file:
    lines = file.readlines()

print(train_list[0])
print(train_list[55])
print(lines[50])