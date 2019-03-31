"""
ファイルを読み込んで文数ごとのファイルへ分けて保存する
"""
import argparse
import os
from tqdm import tqdm


def dic_init(num):
    dic = {}
    for i in range(1, num + 1):
        dic[i] = []

    return dic


def file_check(dir_path):
    if os.path.isdir(dir_path):
        raise IsADirectoryError('{} direcrory is exist'.format(dir_path))
    else:
        os.mkdir(dir_path)


def file_write(sentence_dic, ans_sentence_dic, dir_path):
    for k, v in sentence_dic.items():
        if v == []:
            continue
        num = '0' + str(k) if k < 10 else str(k)
        with open(dir_path + 'q' + num + '.txt', 'a')as f:
            [f.write(line) for line in v]
    for k, v in ans_sentence_dic.items():
        if v == []:
            continue
        num = '0' + str(k) if k < 10 else str(k)
        with open(dir_path + 'a' + num + '.txt', 'a')as f:
            [f.write(line) for line in v]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("que_file")
    parser.add_argument("ans_file")
    parser.add_argument("dir_path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    que_file = args.que_file
    ans_file = args.ans_file
    dir_path = args.dir_path

    # file_check(dir_path)

    max_size = 100
    sentence_dic = dic_init(max_size)
    ans_sentence_dic = dic_init(max_size)

    with open(que_file, 'r')as f:
        q_data = f.readlines()
    with open(ans_file, 'r')as f:
        a_data = f.readlines()

    for i, (q, a) in enumerate(tqdm(zip(q_data, a_data)), start=1):
        line = q.split('|||')
        sent_num = len(line)
        sentence = '\t'.join(line)

        sentence_dic[sent_num].append(sentence)
        ans_sentence_dic[sent_num].append(''.join(a))

        if i % 100000 == 0:
            file_write(sentence_dic, ans_sentence_dic, dir_path)
            sentence_dic = dic_init(max_size)
            ans_sentence_dic = dic_init(max_size)
            break

    file_write(sentence_dic, ans_sentence_dic, dir_path)


if __name__ == '__main__':
    main()