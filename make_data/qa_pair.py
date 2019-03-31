"""
指定文数・単語数のデータにフィルタリング
python script.py [output_dir]
出力 que_best.txt.s{}w{} ans_best.txt.s{}w{}
"""
import sys
import MeCab
from multiprocessing import Pool
from make_data import preprocessing
from tqdm import tqdm
import os
from collections import Counter


def processing(q, a):
    m = MeCab.Tagger('-Owakati -d /home/lr/machida/mecab-0.996/lib/mecab/dic/mecab-ipadic-neologd')
    m.parse('')

    q = ''.join(q.strip().split(' '))
    a = ''.join(a.strip().split(' '))

    q = preprocessing.remove_repetition(q)
    q_sentences = preprocessing.sentence_split(q)

    a = preprocessing.remove_repetition(a)
    a_sentences = preprocessing.sentence_split(a)

    q_wakati = []
    a_wakati = []
    for sentence in q_sentences:
        s = m.parse(sentence).strip()
        q_wakati.append(s)

    for sentence in a_sentences:
        s = m.parse(sentence).strip()
        a_wakati.append(s)

    return q_wakati, a_wakati


def wrapper_processing(args):
    q_res, a_res = processing(args[0], args[1])
    return [q_res, a_res]


def main():
    max_word = 50
    max_sent = 10
    # q_file = '/home/lr/machida/yahoo/que_best.txt'
    # a_file = '/home/lr/machida/yahoo/ans_best.txt'
    q_file = '/home/lr/machida/yahoo/work/data/original_data/que_best.txt'
    a_file = '/home/lr/machida/yahoo/work/data/original_data/ans_best.txt'


    args = sys.argv
    output_dir = args[1]

    if args[2] is not None:
        max_word = 1000
        max_sent = 1000

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(q_file, 'r')as f:
        que_data = f.readlines()
    with open(a_file, 'r')as f:
        ans_data = f.readlines()

    pair = []
    for q, a in zip(que_data, ans_data):
        pair.append([q, a])

    que_word_lit = []
    ans_word_lit = []
    p = Pool()
    res = p.map(wrapper_processing, pair)
    with open(output_dir + 'que_best.txt.s{}.w{}'.format(max_sent, max_word), 'w')as q_f:
        with open(output_dir + 'ans_best.txt.s{}.w{}'.format(max_sent, max_word), 'w')as a_f:
            for r in tqdm(res):
                flag = True
                # max_sent文以上のものはカット
                if len(r[0]) > max_sent or len(r[1]) > max_sent:
                    continue
                # 質問文中に2語以下max_word語以上のものがある場合はカット, '|', '-'が含まれている文もカット
                for rr in r[0]:
                    que_word_lit.append(len(rr.split(' ')))
                    if 2 >= len(rr.split(' ')) or max_word < len(rr.split(' ')) or '|' in rr or '-' in rr:
                        flag = False
                # 回答文中に2語以下max_word語以上のものがある場合はカット
                for rr in r[1]:
                    ans_word_lit.append(len(rr.split(' ')))
                    if 2 >= len(rr.split(' ')) or max_word < len(rr.split(' ')):
                        flag = False
                if flag:
                    q_f.write('|||'.join(r[0]) + '\n')
                    a_f.write('|||'.join(r[1]) + '\n')

    # 統計情報を保存
    que_word_counter = Counter(que_word_lit)
    ans_word_counter = Counter(ans_word_lit)
    with open('que_word_distribution.txt', 'w')as f:
        [f.write('{},{}\n'.format(k, v)) for k, v in sorted(que_word_counter.items())]
    with open('ans_word_distribution.txt', 'w')as f:
        [f.write('{},{}\n'.format(k, v)) for k, v in sorted(ans_word_counter.items())]
    print('Finish')


if __name__ == '__main__':
    main()
