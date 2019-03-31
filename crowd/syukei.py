import sys
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
'''
2
ランサーズのcsvファイルを読み込んで
文番号,重要文番号
になったcsvファイルを出力
出力ファイル名：末尾に.resが付く
'''

def main():
    args = sys.argv

    df = pd.read_csv(args[1])
    df = df.drop(['lancersTaskId', 'lancersResultId', 'lancersNickname', 'set', 'url', 'html_no', 'opinion'], axis=1)

    res = []
    # 割れた事例も集計する場合に使う変数
    # res_index = {}
    # q_num = len(df['no'].unique()) * 50

    # 何文目が正解になっているかをカウントする
    sent_dic = defaultdict(int)

    sent_minus = []
    sent_split = []

    for i in tqdm(range(1, 50 + 1)):
        q = 'q{}'.format(i)
        d = df.groupby('no')[q].value_counts().reset_index(name='counts')
        temp_dic = {}
        for index, row in d.iterrows():
            if row[q] != -1 and row['counts'] >= 4:
                res.append([(row['no']-1) * 50 + i, row[q]])
                # res_index[(row['no']-1) * 50 + i] = row[q]
                sent_dic[row[q]] += 1

            temp_dic[row[q]] = row['counts']
            if sum(temp_dic.values()) == 5:
                if -1 in temp_dic:
                    if temp_dic[-1] >= 4:
                        sent_minus.append((row['no']-1) * 50 + i)
                    else:
                        sent_no, count = max(temp_dic.items(), key=lambda x: x[1])
                        if count < 4:
                            sent_split.append((row['no']-1) * 50 + i)
                else:
                    sent_no, count = max(temp_dic.items(), key=lambda x: x[1])
                    if count < 4:
                        sent_split.append((row['no'] - 1) * 50 + i)
                temp_dic = {}

    res.sort()

    # 割れた事例も含める
    # with open(args[1] + '.res_total', 'w')as f:
    #     f.write('que_no,sent_no\n')
    #     for i in range(1, q_num + 1):
    #         if i in res_index:
    #             f.write('{},{}\n'.format(i, res_index[i]))
    #         else:
    #             f.write('{},{}\n'.format(i, 0))

    with open(args[1] + '.res', 'w')as f:
        f.write('que_no,sent_no\n')
        [f.write('{},{}\n'.format(r[0], r[1])) for r in res]

    print(sent_dic)
    print('correct: {}, minus: {}, split: {}'.format(len(res), len(sent_minus), len(sent_split)))

    # デバック用
    # for name, item in df.iteritems():
    #     print(name)
    #     a = item.value_counts()
    #     for k,v in a.items():
    #         print(k, v)
    #
    #     print(item[0])
    #     print(item[1])
    #     exit()


if __name__ == '__main__':
    main()