"""
queファイルを石垣さんモデルに合わせたファイルを出力
ここでは先頭に文数を表示するようにした
python script.py que_best.txt.s{}w{}
出力 que_best.txt.s{}w{}.dist
"""
import sys
from tqdm import tqdm


def main():
    args = sys.argv
    que_file_name = args[1]

    data = []
    with open(que_file_name, 'r')as f:
        que = f.readlines()

    for line in tqdm(que):
        bunsu = len(line.split('|||'))
        line = '{}\t'.format(bunsu) + line.replace('|||', '\t').replace(' ', '')
        data.append(line)

    with open(que_file_name + '.dist', 'w')as f:
        [f.write(d) for d in data]


main()