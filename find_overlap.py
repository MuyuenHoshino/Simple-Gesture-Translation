



with open("./daily_real_balanced_map.txt", 'r') as file:
    lines = file.readlines()


train_list = []

for line in lines:
    if line.split()[0] not in train_list:
        train_list.append(line[0:6])

train_list.append("000058")

with open("./dataset_real_val.txt", 'r') as file:
    lines = file.readlines()


val_list = []

for line in lines:
    val_list.append(line[0:-1])



with open("./dataset_real_test.txt", 'r') as file:
    lines = file.readlines()


test_list = []

for line in lines:
    test_list.append(line[0:-1])

print(train_list[0])
print(test_list[0])

common_elements = set(train_list) & set(val_list)
result = list(common_elements)

print(result)