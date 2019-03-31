import sys
from tqdm import tqdm
from collections import defaultdict
import random

def main():
    train_size = 100000
    valid_size = 10000

    args = sys.argv
    que_file = args[1]
    ans_file = args[2]

    with open(que_file, 'r')as f:
        q_data = f.readlines()
    with open(ans_file, 'r')as f:
        a_data = f.readlines()

    dic = defaultdict(int)
    random.seed(0)
    q_res = random.sample(q_data, train_size + valid_size)
    random.seed(0)
    a_res = random.sample(a_data, train_size + valid_size)

    for q in q_res:
        sent_num = len(q.split('\t')[1].split('|||'))
        dic[sent_num] += 1
    print(dic)

    with open(que_file + '.sample.train', 'w')as f:
        [f.write(r) for r in q_res[:train_size]]
    with open(ans_file + '.sample.train', 'w')as f:
        [f.write(r) for r in a_res[:train_size]]
    with open(que_file + '.sample.valid', 'w')as f:
        [f.write(r) for r in q_res[train_size:]]
    with open(ans_file + '.sample.valid', 'w')as f:
        [f.write(r) for r in a_res[train_size:]]

main()