import sys
import re
import evaluate
from itertools import zip_longest
from collections import Counter
import numpy as np

np.random.seed(1)


class GridSearch:
    def __init__(self, correct_txt_file, valid_num=5):
        with open(correct_txt_file, 'r')as f:
            self.correct_data = f.readlines()
        self.valid_num = len(self.correct_data) // valid_num
        self.ev = evaluate.Evaluate(correct_txt_file)
        
        self.result_log = []

    def slice(self, lit):
        valid_num = self.valid_num
        if valid_num == 0:
            return [lit]
        res = []
        for a in zip_longest(*[iter(lit)] * valid_num):
            res.append(list(a))
        return res

    def split_dev_test(self, lit, index):
        dev, test = [], []
        for i, l in enumerate(lit):
            if i == index:
                test = l
            else:
                dev.extend(l)
        return dev, test

    def shuffle_list(self, *args):
        zipped = list(zip(*args))
        np.random.shuffle(zipped)
        shuffled_list = zip(*zipped)
        return shuffled_list

    def split_data(self, label_list, align_list=None, detail_flag=False):
        param = []
        total = [0, 0, 0, 0, 0, 0, 0]
        detail = []

        if align_list:
            c_data, l_data, a_data = self.shuffle_list(self.correct_data, label_list, align_list)
            align = self.slice(a_data)
        else:
            c_data, l_data = self.shuffle_list(self.correct_data, label_list)
        
        correct = self.slice(c_data)
        label = self.slice(l_data)

        for i in range(len(correct)):
            c_dev, c_test = self.split_dev_test(correct, i)
            l_dev, l_test = self.split_dev_test(label, i)
            a_dev, a_test = [], []
            if align_list:
                a_dev, a_test = self.split_dev_test(align, i)

            self.ev.correct_data = c_dev
            best_param_dic = self.ev.param_search(l_dev, a_dev)
            k = max(best_param_dic, key=lambda x: best_param_dic[x])
            v = best_param_dic[k]
            param.append(k)
            detail.append('{} dev: {} {}'.format(i + 1, k, v))
            self.ev.correct_data = c_test
            k = k.split(' ')
            if len(k) == 1:
                s_rate, s_count, m_rate, m_count = self.ev.label(l_test)
            elif len(k) == 2:
                if k[0] == 'init':
                    s_rate, s_count, m_rate, m_count = self.ev.label_init(l_test, float(k[1]))
                else:
                    s_rate, s_count, m_rate, m_count = self.ev.label_mix_align(l_test, a_test, float(k[1]))
            else:
                s_rate, s_count, m_rate, m_count = self.ev.label_mix_aligh_init(l_test, a_test, float(k[1]), float(k[3]))
            detail.append(' test: {}'.format(' '.join(s_rate)))
            total = [x + float(y) for x, y in zip(total, s_rate)]

        s_total = round(total[-1], 3)
        total = ' '.join([str(round(t / len(correct), 3)) for t in total])
        detail.append('total: {}'.format(total))

        if detail_flag:
            for d in detail:
                print(d)

        # パラメータ, スコア, スコアの合計
        return ['|'.join(param), total, s_total]


def parse_param(param):
    init = -1
    mix  = -1
    param = param.split(' ')
    # normal
    if len(param) == 1:
        return init, mix
    # init or mix
    if len(param) == 2:
        if param[0] == 'init':
            init = float(param[1])
            return init, mix
        elif param[0] == 'mix':
            mix = float(param[1])
            return init, mix
    # init and mix
    init = float(param[1])
    mix = float(param[3])

    return init, mix


if __name__ == '__main__':
    args = sys.argv
    model_name = args[1]
    model_dir = re.search(r'^(.*/)', args[1]).group(1)

    label, align, correct = evaluate.load_score_file(model_name, model_dir)

    gs = GridSearch(correct)
    res = gs.split_data(label, align, detail_flag=True)
    c_res = Counter(res[0].split('|'))
    param = max(c_res, key=lambda x: c_res[x])
    init, mix = parse_param(param)

    s_rate, s_count, m_rate, m_count = evaluate.eval_param(model_name, label, align, correct, init, mix)
    print(param)
    print('s: {} | {}'.format(' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    # print('m: {} | {}'.format(' '.join(x for x in m_rate), ' '.join(x for x in m_count)))