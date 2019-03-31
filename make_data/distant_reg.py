"""
distant scoreファイルからデータの形を合わせる
python script.py [que_best.txt.s{}w{}] [score.txt] [ans_best.txt.s{}w{}]
出力 que_best.txt.s{}w{}.label ans_best.txt.s{}w{}.label
"""
import sys


def distant_label(file_name):
    with open(file_name, 'r')as f:
        data = f.read().strip('-').strip().split('\n--------\n')
    score = []

    for d in data:
        s = []
        for i, dd in enumerate(d.split('\n'), start=1):
            s.append(str(dd.split('\t')[0]))
        score.append(','.join(s))

    return score


def main():
    args = sys.argv
    split_que_file = args[1]
    distant_score_file = args[2]
    split_ans_file = args[3]

    score_label = distant_label(distant_score_file)

    with open(split_que_file, 'r')as f:
        data = f.readlines()

    res = []
    for l, d in zip(score_label, data):
        res.append('{}\t{}'.format(l, d))

    with open(split_que_file + '.label', 'w')as f:
        [f.write(r) for r in res]

    with open(split_ans_file, 'r')as f:
        ans = f.readlines()

    data = []
    for line in ans:
        line = line.replace('|||', ' ')
        data.append(line)

    with open(split_ans_file + '.label', 'w')as f:
        [f.write(d) for d in data]
    print('que: {}, ans: {}'.format(len(res), len(data)))


main()
