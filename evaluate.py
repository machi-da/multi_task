import argparse
import configparser
import re
import os
import glob
import copy
import numpy as np

import dataset


class Evaluate:
    def save_single_result(self, file_name, init, mix):
        print('save single result: {}'.format(file_name))
        key = param_to_key(init, mix)
        with open(file_name, 'w')as f:
            for i, r in zip(self.single_index, self.single_result_dic[key].values()):
                f.write('{}:\t{}\n'.format(i, r))
        return

    def label(self, label_list, correct_label_list, correct_index_list, multi_sentence_score=False):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, clabel in zip(label_data, correct_label_list):
            rank = []
            for _ in range(len(label)):
                index = label.argmax()
                if index in clabel:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1000
            rank_list.append(rank)

        s_rate, s_count, s_result = self.single(rank_list, correct_index_list)
        m_rate, m_count = [], []

        if multi_sentence_score:
            m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count, s_result

    def label_init(self, label_list, correct_label_list, correct_index_list, init_threshold=0.7, multi_sentence_score=False):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, clabel in zip(label_data, correct_label_list):
            rank = []
            true_index = list(np.where(label >= init_threshold)[0])
            for _ in range(len(label)):
                index = label.argmax()

                # 先頭を優先
                if len(true_index) > 0:
                    if index != true_index[0]:
                        if true_index[0] in clabel:
                            rank.append((true_index[0], True))
                        else:
                            rank.append((true_index[0], False))
                        label[true_index[0]] = -1000
                        true_index = true_index[1:]
                        continue

                if index in clabel:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1000
                if index in true_index:
                    true_index.remove(index)
            rank_list.append(rank)

        s_rate, s_count, s_result = self.single(rank_list, correct_index_list)
        m_rate, m_count = [], []

        if multi_sentence_score:
            m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count, s_result

    def label_mix_align(self, label_list, align_list, correct_label_list, correct_index_list, weight=0.5, multi_sentence_score=False):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, clabel, align in zip(label_data, correct_label_list, align_list):
            label = weight * label + (1 - weight) * align
            rank = []
            for _ in range(len(label)):
                index = label.argmax()

                if index in clabel:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1000
            rank_list.append(rank)

        s_rate, s_count, s_result = self.single(rank_list, correct_index_list)
        m_rate, m_count = [], []

        if multi_sentence_score:
            m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count, s_result

    def label_mix_aligh_init(self, label_list, align_list, correct_label_list, correct_index_list, init_threshold=0.7, weight=0.5, multi_sentence_score=False):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, clabel, align in zip(label_data, correct_label_list, align_list):
            label = weight * label + (1 - weight) * align
            rank = []
            true_index = list(np.where(label >= init_threshold)[0])
            for _ in range(len(label)):
                index = label.argmax()

                # 先頭を優先
                if len(true_index) > 0:
                    if index != true_index[0]:
                        if true_index[0] in clabel:
                            rank.append((true_index[0], True))
                        else:
                            rank.append((true_index[0], False))
                        label[true_index[0]] = -1000
                        true_index = true_index[1:]
                        continue

                if index in clabel:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1000
                if index in true_index:
                    true_index.remove(index)
            rank_list.append(rank)

        s_rate, s_count, s_result = self.single(rank_list, correct_index_list)
        m_rate, m_count = [], []

        if multi_sentence_score:
            m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count, s_result

    def single(self, rank_list, correct_index_list):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        result = []

        if correct_index_list:
            for index, (r, cindex) in enumerate(zip(rank_list, correct_index_list), start=1):
                sent_num = len(r)
                # 正解ラベルの数: correct_num
                count_num = 0
                for rr in r:
                    if rr[1]:
                        count_num += 1

                if count_num == 1 or count_num == 0:
                    score_dic[sent_num][1] += 1
                    if r[0][1]:  # 一番スコアの高い文(rの0番目)が正解ラベル判定でTrueかどうか
                        score_dic[sent_num][0] += 1
                        result.append([cindex, 'T'])
                    else:
                        result.append([cindex, 'F'])

        else:
            for index, r in enumerate(rank_list, start=1):
                sent_num = len(r)
                # 正解ラベルの数: correct_num
                count_num = 0
                for rr in r:
                    if rr[1]:
                        count_num += 1

                if count_num == 1 or count_num == 0:
                    score_dic[sent_num][1] += 1
                    if r[0][1]:  # 一番スコアの高い文(rの0番目)が正解ラベル判定でTrueかどうか
                        score_dic[sent_num][0] += 1

        t_correct, t = sum([v[0] for k, v in score_dic.items()]), sum([v[1] for k, v in score_dic.items()])
        for v in score_dic.values():
            if v[1] == 0:
                v[1] = 1
        rate = [str(round(v[0] / v[1], 3)) for k, v in score_dic.items()]
        micro_rate = sum([v[0] / v[1] for k, v in score_dic.items()]) / len(rate)
        rate.append(str(round(micro_rate, 3)))
        count = ['{}/{}'.format(v[0], v[1]) for k, v in score_dic.items()]
        count.append('{}/{}'.format(t_correct, t))

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

    def param_search(self, label_list, align_list, correct_label_list):
        best_param_dic = {}

        s_rate, _, _, _, _ = self.label(label_list, correct_label_list, [])
        key = 'normal'
        best_param_dic[key] = s_rate[-1]

        # range: 1~9
        for i in range(1, 10):
            init_threshold = round(i * 0.1, 1)
            s_rate, _, _, _, _ = self.label_init(label_list, correct_label_list, [], init_threshold)
            key = 'init {}'.format(init_threshold)
            best_param_dic[key] = s_rate[-1]

        if align_list:
            for i in range(1, 10):
                weight = round(i * 0.1, 1)
                s_rate, _, _, _, _ = self.label_mix_align(label_list, align_list, correct_label_list, [], weight)
                key = 'mix {}'.format(weight)
                best_param_dic[key] = s_rate[-1]

            for i in range(1, 10):
                weight = round(i * 0.1, 1)
                for j in range(1, 10):
                    init_threshold = round(j * 0.1, 1)
                    s_rate, _, _, _, _ = self.label_mix_aligh_init(label_list, align_list, correct_label_list, [], init_threshold, weight)
                    key = 'init {} mix {}'.format(init_threshold, weight)
                    best_param_dic[key] = s_rate[-1]

        return best_param_dic

    def eval_param(self, label_list, align_list, correct_label_list, correct_index_list, init=-1, mix=-1):
        if init == -1:
            if mix == -1:
                s_rate, s_count, m_rate, m_count, s_result = self.label(label_list, correct_label_list, correct_index_list)
            else:
                s_rate, s_count, m_rate, m_count, s_result = self.label_mix_align(label_list, align_list, correct_label_list, correct_index_list, mix)
        else:
            if mix == -1:
                s_rate, s_count, m_rate, m_count, s_result = self.label_init(label_list, correct_label_list, correct_index_list, init)
            else:
                s_rate, s_count, m_rate, m_count, s_result = self.label_mix_aligh_init(label_list, align_list, correct_label_list, correct_index_list, init, mix)

        return s_rate, s_count, m_rate, m_count, s_result


def param_to_key(init, mix):
    if init == -1:
        if mix == -1:
            return 'normal'
        else:
            return 'mix {}'.format(mix)
    else:
        if mix == -1:
            return 'init {}'.format(init)
        else:
            return 'init {} mix {}'.format(init, mix)


def key_to_param(key):
    k = key.split(' ')

    if len(k) == 1:
        return -1, -1
    elif len(k) == 2:
        if k[0] == 'init':
            return float(k[1]), -1
        else:
            return -1, float(k[1])
    else:
        return float(k[1]), float(k[3])


def load_score_file(model_name, model_dir):
    label = []
    align = []
    correct_label = []
    correct_index = []

    label_file = model_name + '.label'
    if os.path.isfile(label_file):
        label = dataset.load_score_file(label_file)

    align_file = model_name + '.align'
    if os.path.isfile(align_file):
        align = dataset.load_score_file(align_file)

    config = configparser.ConfigParser()
    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    if len(config_files):
        config_file = config_files[0]
        config.read(config_file)
        model_info = model_dir.strip('/').split('_')
        data_path = 'local' if 'l' in model_info else 'server'
        if 'super' in model_info:
            correct = config[data_path]['single_src_file']
        else:
            correct = config[data_path]['test_src_file']
        correct_label, _, correct_index = dataset.load_with_label_index(correct)

        if 'encdec' in model_dir.split('_'):
            raw_data = config[data_path]['raw_score_file']
            label = dataset.load_score_file(raw_data)

    return label, align, correct_label, correct_index


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

    label, align, correct_label, correct_index = load_score_file(model_name, model_dir)
    ev = Evaluate()
    s_rate, s_count, m_rate, m_count, s_result = ev.eval_param(label, align, correct_label, correct_index, init, mix)

    print('s: {} | {}'.format(' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    # print('m: {} | {}'.format(' '.join(x for x in m_rate), ' '.join(x for x in m_count)))