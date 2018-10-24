import argparse
import configparser
import re
import os
import glob
import copy
import numpy as np

import dataset


class Evaluate:
    def __init__(self, correct_txt_file):
        self.correct_label = []
        with open(correct_txt_file, 'r')as f:
            correct_data = f.readlines()
        for d in correct_data:
            self.correct_label.append([int(num) - 1 for num in d.split('\t')[0].split(',')])

        single_index = ['method', 'score']
        for i, d in enumerate(correct_data, start=1):
            correct_label = d.split('\t')[0].split(',')
            if len(correct_label) == 1 and correct_label[0] != '0':
                single_index.append(str(i))
        self.single_index = ','.join(single_index)
        self.single_result = None

    def save_single_result(self, file_name):
        print('save single result: {}'.format(file_name))
        with open(file_name, 'w')as f:
            [f.write(r + '\n') for r in self.single_result]
        return

    def label(self, label_list):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, correct in zip(label_data, self.correct_label):
            rank = []
            for _ in range(len(label)):
                index = label.argmax()
                if index in correct:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1000
            rank_list.append(rank)

        method = 'normal'
        s_rate, s_count, s_result = self.single(rank_list, method)
        self.single_result = [self.single_index, s_result]
        m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count

    def label_init(self, label_list, init_threshold=0.7):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, correct in zip(label_data, self.correct_label):
            rank = []
            true_index = list(np.where(label >= init_threshold)[0])
            for _ in range(len(label)):
                index = label.argmax()

                # 先頭を優先
                if len(true_index) > 0:
                    if index != true_index[0]:
                        if true_index[0] in correct:
                            rank.append((true_index[0], True))
                        else:
                            rank.append((true_index[0], False))
                        label[true_index[0]] = -1000
                        true_index = true_index[1:]
                        continue

                if index in correct:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1000
                if index in true_index:
                    true_index.remove(index)
            rank_list.append(rank)

        method = 'init_{}'.format(init_threshold)
        s_rate, s_count, s_result = self.single(rank_list, method)
        self.single_result = [self.single_index, s_result]
        m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count

    def label_mix_align(self, label_list, align_list, weight=0.5):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, correct, align in zip(label_data, self.correct_label, align_list):
            label = weight * label + (1 - weight) * align
            rank = []
            for _ in range(len(label)):
                index = label.argmax()

                if index in correct:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1000
            rank_list.append(rank)

        method = 'mix_{}'.format(weight)
        s_rate, s_count, s_result = self.single(rank_list, method)
        self.single_result = [self.single_index, s_result]
        m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count

    def label_mix_aligh_init(self, label_list, align_list, init_threshold=0.7, weight=0.5):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, correct, align in zip(label_data, self.correct_label, align_list):
            label = weight * label + (1 - weight) * align
            # label = label * align
            rank = []
            # threshold = init_threshold + (len(label) - 1)
            true_index = list(np.where(label >= init_threshold)[0])
            for _ in range(len(label)):
                index = label.argmax()

                # 先頭を優先
                if len(true_index) > 0:
                    if index != true_index[0]:
                        if true_index[0] in correct:
                            rank.append((true_index[0], True))
                        else:
                            rank.append((true_index[0], False))
                        label[true_index[0]] = -1000
                        true_index = true_index[1:]
                        continue

                if index in correct:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1000
                if index in true_index:
                    true_index.remove(index)
            rank_list.append(rank)

        method = 'init_{} mix_{}'.format(init_threshold, weight)
        s_rate, s_count, s_result = self.single(rank_list, method)
        self.single_result = [self.single_index, s_result]
        m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count

    def single(self, rank_list, method):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        result = [method]
        # sentence_num = ['-', '-']
        for index, r in enumerate(rank_list, start=1):
            sent_num = len(r)
            # 正解ラベルの数: correct_num
            count_num = 0
            for rr in r:
                if rr[1]:
                    count_num += 1
            # 正解した数: correct
            correct = 0
            for i in range(count_num):
                if r[i][1]:
                    correct += 1
            if count_num == 1 or count_num == 0:
                score_dic[sent_num][1] += 1
                if correct:
                    score_dic[sent_num][0] += 1
                    result.append('1')
                else:
                    result.append('0')
                # sentence_num.append(str(sent_num))

        t_correct, t = sum([v[0] for k, v in score_dic.items()]), sum([v[1] for k, v in score_dic.items()])
        for v in score_dic.values():
            if v[1] == 0:
                v[1] = 1
        rate = [str(round(v[0] / v[1], 3)) for k, v in score_dic.items()]
        micro_rate = sum([v[0] / v[1] for k, v in score_dic.items()]) / len(rate)
        rate.append(str(round(micro_rate, 3)))
        count = ['{}/{}'.format(v[0], v[1]) for k, v in score_dic.items()]
        count.append('{}/{}'.format(t_correct, t))

        result.insert(1, str(rate[-1]))
        result = ','.join(result)
        # sentence_num = ','.join(sentence_num)
        # self.single_result.append(sentence_num)

        return rate, count, result

    def multiple(self, rank_list):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        for r in rank_list:
            sent_num = len(r)
            count_num = 0
            for rr in r:
                if rr[1]:
                    count_num += 1
            correct = 0
            for i in range(count_num):
                if r[i][1]:
                    correct += 1

            score_dic[sent_num][0] += correct
            score_dic[sent_num][1] += count_num

        t_correct, t = sum([v[0] for k, v in score_dic.items()]), sum([v[1] for k, v in score_dic.items()])
        for v in score_dic.values():
            if v[1] == 0:
                v[1] = 1
        rate = [str(round(v[0] / v[1], 3)) for k, v in score_dic.items()]
        micro_rate = sum([v[0] / v[1] for k, v in score_dic.items()]) / len(rate)
        rate.append(str(round(micro_rate, 3)))
        count = ['{}/{}'.format(v[0], v[1]) for k, v in score_dic.items()]
        count.append('{}/{}'.format(t_correct, t))
        return rate, count

    def param_search(self, label_list, align_list):
        best_param_dic = {}

        s_rate, _, _, _ = self.label(label_list)
        key = 'normal'
        best_param_dic[key] = s_rate[-1]
        # print('{}\t{}'.format(key, ' '.join(s_rate)))

        # range: 1~9
        for i in range(1, 10):
            init_threshold = round(i * 0.1, 1)
            s_rate, _, _, _ = self.label_init(label_list, init_threshold)
            key = 'init {}'.format(init_threshold)
            best_param_dic[key] = s_rate[-1]
            # print('{}\t{}'.format(key, ' '.join(s_rate)))

        if align_list:
            for i in range(1, 10):
                weight = round(i * 0.1, 1)
                s_rate, _, _, _ = self.label_mix_align(label_list, align_list, weight)
                key = 'mix {}'.format(weight)
                best_param_dic[key] = s_rate[-1]
                # print('{}\t{}'.format(key, ' '.join(s_rate)))

            for i in range(1, 10):
                weight = round(i * 0.1, 1)
                for j in range(1, 10):
                    init_threshold = round(j * 0.1, 1)
                    s_rate, _, _, _ = self.label_mix_aligh_init(label_list, align_list, init_threshold, weight)
                    key = 'init {} mix {}'.format(init_threshold, weight)
                    best_param_dic[key] = s_rate[-1]
                    # print('{}\t{}'.format(key, ' '.join(s_rate)))

        return best_param_dic


def eval_param(model_name, label, align, correct, init=-1, mix=-1):
    evaluater = Evaluate(correct)
    if init == -1:
        if mix == -1:
            s_rate, s_count, m_rate, m_count = evaluater.label(label)
        else:
            s_rate, s_count, m_rate, m_count = evaluater.label_mix_align(label, align, mix)
    else:
        if mix == -1:
            s_rate, s_count, m_rate, m_count = evaluater.label_init(label, init)
        else:
            s_rate, s_count, m_rate, m_count = evaluater.label_mix_aligh_init(label, align, init, mix)

    evaluater.save_single_result(model_name + '.s_res.csv')
    return s_rate, s_count, m_rate, m_count


def load_score_file(model_name, model_dir):
    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    config_file = config_files[0]
    config = configparser.ConfigParser()
    config.read(config_file)

    data_path = 'local' if model_dir.split('_')[2] == 'l' else 'server'
    correct = config[data_path]['test_src_file']


    # correct = '/Users/machida/work/yahoo/util/correct1-2.txt'

    label = []
    align = []
    if model_dir.split('_')[0] == 'encdec':
        raw_data = config[data_path]['raw_score_file']
        label = dataset.txt_to_list(raw_data)
    else:
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

    return label, align, correct


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--init', '-i', type=float, default=-1)
    parser.add_argument('--mix', '-m', type=float, default=-1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name
    init = args.init
    mix = args.mix
    model_dir = re.search(r'^(.*/)', model_name).group(1)

    label, align, correct = load_score_file(model_name, model_dir)
    s_rate, s_count, m_rate, m_count = eval_param(model_name, label, align, correct, init, mix)

    print('s: {} | {}'.format(' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    # print('m: {} | {}'.format(' '.join(x for x in m_rate), ' '.join(x for x in m_count)))