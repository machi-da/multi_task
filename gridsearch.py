import argparse
import re
import evaluate
from itertools import zip_longest
import numpy as np


class GridSearch:
    def __init__(self, valid_num):
        self.ev = evaluate.Evaluate()
        self.valid_num = valid_num

        self.result_detail = None

    def print_detail(self):
        for r in self.result_detail:
            print(r)

    def gridsearch(self, correct_label_list, correct_index_list, label_list, align_list=None):
        self.result_detail = []
        param = []
        total = [0, 0, 0, 0, 0, 0, 0]

        slice_size = len(correct_label_list) // self.valid_num

        if align_list:
            c_data, ci_data, l_data, a_data = shuffle_list(correct_label_list, correct_index_list, label_list, align_list)
            align = slice_list(a_data, slice_size)
        else:
            c_data, ci_data, l_data = shuffle_list(correct_label_list, correct_index_list, label_list)

        correct = slice_list(c_data, slice_size)
        correct_index = slice_list(ci_data, slice_size)
        label = slice_list(l_data, slice_size)

        dev_score = []
        s_result_total = []
        for i in range(len(correct)):
            c_dev, c_test = split_dev_test(correct, i)
            ci_dev, ci_test = split_dev_test(correct_index, i)
            l_dev, l_test = split_dev_test(label, i)
            a_dev, a_test = [], []
            if align_list:
                a_dev, a_test = split_dev_test(align, i)
            best_param_dic = self.ev.param_search(l_dev, a_dev, c_dev)
            k = max(best_param_dic, key=lambda x: best_param_dic[x])
            v = best_param_dic[k]
            param.append(k)
            dev_score.append(float(v))
            self.result_detail.append('{} dev: {} {}'.format(i + 1, k, v))
            init, mix = evaluate.key_to_param(k)
            s_rate, s_count, m_rate, m_count, s_result = self.ev.eval_param(l_test, a_test, c_test, ci_test, init, mix)
            s_result_total.extend(s_result)

            self.result_detail.append(' test: {}'.format(' '.join(s_rate)))
            total = [x + float(y) for x, y in zip(total, s_rate)]

        dev_score = sum(dev_score) / len(correct)
        s_total = total[-1] / len(correct)
        total = ' '.join([str(round(t / len(correct), 3)) for t in total])
        self.result_detail.append('total: {}'.format(total))

        # パラメータ, スコア, スコアの平均, スコアの詳細, devスコア
        return '|'.join(param), total, s_total, s_result_total, dev_score


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
    np.random.seed(1)
    zipped = list(zip(*args))
    np.random.shuffle(zipped)
    shuffled_list = zip(*zipped)
    return shuffled_list


def main(label, align, correct_label, correct_index, valid_num, align_only=False, print_flag=True):
    gs = GridSearch(valid_num=valid_num)

    if align_only:
        param, total, s_total, s_result, dev_score = gs.gridsearch(correct_label, correct_index, align, [])
    else:
        param, total, s_total, s_result, dev_score = gs.gridsearch(correct_label, correct_index, label, align)

    if print_flag:
        gs.print_detail()

    return s_total, s_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--valid_num', '-v', type=int, default=5)
    parser.add_argument('--align_only', '-a', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    model_dir = re.search(r'^(.*/)', model_name).group(1)
    valid_num = args.valid_num
    align_only = args.align_only

    label, align, correct_label, correct_index = evaluate.load_score_file(model_name, model_dir)

    main(label, align, correct_label, correct_index, valid_num, align_only)