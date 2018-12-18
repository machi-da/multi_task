import sys

args = sys.argv
dir_path = args[1]

valid_index = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: []
}
with open('valid.txt', 'r')as f:
    data = f.readlines()
    for d in data[1:]:
        d = d.strip().split(',')
        valid_index[1].append(int(d[0]))
        valid_index[2].append(int(d[1]))
        valid_index[3].append(int(d[2]))
        valid_index[4].append(int(d[3]))
        valid_index[5].append(int(d[4]))

temp = []
for i in range(1, 6):
    with open(dir_path + 'valid{}/model_epoch_{}.label'.format(i, args[i + 1]), 'r')as f:
        data = f.readlines()
    for index, d in zip(valid_index[i], data):
        temp.append([index, d])

with open(dir_path + 'valid_concat.label', 'w')as f:
    for t in sorted(temp, key=lambda x: x[0]):
        f.write(t[1])