import configparser
import sys
import re
import glob
import os
import evaluate
from itertools import zip_longest
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

    def log(self, log_text):
        self.result_log.append(log_text)
        print(log_text)
        return

    def save_log(self, file_name):
        with open(file_name + '.grid', 'r')as f:
            [f.write(r + '\n') for r in self.result_log]
        return

    def split_data(self, label_list, align_list=None):
        if align_list:
            c_data, l_data, a_data = self.shuffle_list(self.correct_data, label_list, align_list)
            align = self.slice(a_data)
        else:
            c_data, l_data = self.shuffle_list(self.correct_data, label_list)
        
        correct = self.slice(c_data)
        label = self.slice(l_data)

        total = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(correct)):
            c_dev, c_test = self.split_dev_test(correct, i)
            l_dev, l_test = self.split_dev_test(label, i)
            a_dev, a_test = [], []
            if align_list:
                a_dev, a_test = self.split_dev_test(align, i)

            self.ev.correct_data = c_dev
            best_param_dic = self.ev.param_search(file_name, l_dev, a_dev)
            for k, v in sorted(best_param_dic.items(), key=lambda x: x[1], reverse=True)[:1]:
                self.log('{} dev: {} {}'.format(i + 1, k, v))
                self.ev.correct_data = c_test
                k = k.split(' ')
                if len(k) == 1:
                    s_rate, s_count, m_rate, m_count, _ = self.ev.label(l_test)
                elif len(k) == 2:
                    if k[0] == 'init':
                        s_rate, s_count, m_rate, m_count, _ = self.ev.label_init(l_test, float(k[1]))
                    else:
                        s_rate, s_count, m_rate, m_count, _ = self.ev.label_mix_align(l_test, a_test, float(k[1]))
                else:
                    s_rate, s_count, m_rate, m_count, _ = self.ev.label_mix_aligh_init(l_test, a_test, float(k[1]), float(k[3]))
                self.log(' test: {}'.format(' '.join(s_rate)))
                total = [x + float(y) for x, y in zip(total, s_rate)]
        total = [str(round(t / len(correct), 3)) for t in total]
        self.log(' '.join(total))


if __name__ == '__main__':
    args = sys.argv
    model_name = args[1][:-4]
    model_dir = re.search(r'^(.*/)', args[1]).group(1)

    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    config_file = config_files[0]
    config = configparser.ConfigParser()
    config.read(config_file)

    data_type = model_dir.split('_')[2]
    section = evaluate.model_type(data_type)
    correct = config[section]['test_src_file']

    label = []
    align = []
    with open(model_name + '.label', 'r')as f:
        label_data = f.readlines()
    for line in label_data:
        line = line[1:-2]
        score = np.array([float(l) for l in line.split()])
        label.append(score)
    is_align = os.path.isfile(model_name + '.align')

    if is_align:
        with open(model_name + '.align', 'r')as f:
            attn_data = f.readlines()
        for line in attn_data:
            line = line[1:-2]
            score = np.array([float(l) for l in line.split()])
            align.append(score)

    gs = GridSearch(correct)
    gs.split_data(label, align)
    gs.save_log(model_name)