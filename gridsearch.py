import sys
import re
import evaluate
from itertools import zip_longest
from collections import Counter
import numpy as np

np.random.seed(1)


class GridSearch:
    def __init__(self, valid_num):
        self.ev = evaluate.Evaluate()
        self.valid_num = valid_num

        self.result_log = []

    def gridsearch(self, correct_label, label_list, align_list=None, detail_flag=False):
        param = []
        total = [0, 0, 0, 0, 0, 0, 0]
        detail = []

        slice_size = len(correct_label) // self.valid_num

        if align_list:
            c_data, l_data, a_data = shuffle_list(correct_label, label_list, align_list)
            align = slice_list(a_data, slice_size)
        else:
            c_data, l_data = shuffle_list(correct_label, label_list)

        correct = slice_list(c_data, slice_size)
        label = slice_list(l_data, slice_size)

        for i in range(len(correct)):
            c_dev, c_test = split_dev_test(correct, i)
            l_dev, l_test = split_dev_test(label, i)
            a_dev, a_test = [], []
            if align_list:
                a_dev, a_test = split_dev_test(align, i)

            self.ev.correct_label = c_dev
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

        s_total = total[-1]
        total = ' '.join([str(round(t / len(correct), 3)) for t in total])
        detail.append('total: {}'.format(total))

        c_param = Counter(param)
        best_param = max(c_param, key=lambda x: c_param[x])
        init, mix = parse_param(best_param)

        if detail_flag:
            for d in detail:
                print(d)

        # パラメータ, スコア, スコアの合計
        return '|'.join(param), total, s_total, init, mix


def slice_list(lit, slice_size):
    if slice_size == 0:
        return [lit]
    res = []
    for a in zip_longest(*[iter(lit)] * slice_size):
        res.append(list(a))
    return res


def split_dev_test(lit, index):
    dev, test = [], []
    for i, l in enumerate(lit):
        if i == index:
            test = l
        else:
            dev.extend(l)
    return dev, test


def split_train_dev_test(lit, index):
    dev_index = index
    test_index = index + 1
    if test_index == len(lit):
        test_index = 0
    train, dev, test = [], [], []
    for i, l in enumerate(lit):
        if i == dev_index:
            dev = l
        elif i == test_index:
            test = l
        else:
            train.extend(l)
    return train, dev, test


def shuffle_list(*args):
    zipped = list(zip(*args))
    np.random.shuffle(zipped)
    shuffled_list = zip(*zipped)
    return shuffled_list


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


def main(model_name, label, align, correct_label, single_index):
    gs = GridSearch(valid_num=5)
    param, total, s_total, init, mix = gs.gridsearch(correct_label, label, align, detail_flag=True)
    # param, total, s_total, init, mix = gs.gridsearch(correct_label, align, [], detail_flag=True)

    ev = evaluate.Evaluate(correct_label, single_index)
    s_rate, s_count, m_rate, m_count = ev.eval_param(model_name, label, align, init, mix)
    print('init {}, mix {}'.format(init, mix))
    print('s: {} | {}'.format(' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    # print('m: {} | {}'.format(' '.join(x for x in m_rate), ' '.join(x for x in m_count)))

    return s_rate


if __name__ == '__main__':
    args = sys.argv
    model_name = args[1]
    model_dir = re.search(r'^(.*/)', args[1]).group(1)

    label, align, correct_label, single_index = evaluate.load_score_file(model_name, model_dir)

    main(model_name, label, align, correct_label, single_index)