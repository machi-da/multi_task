import configparser
import sys
import re
import os
import glob
import copy
import numpy as np


class Evaluate:
    def __init__(self, correct_txt_file):
        with open(correct_txt_file, 'r')as f:
            self.correct_data = f.readlines()

        single_index = ['method', 'score']
        for i, d in enumerate(self.correct_data, start=1):
            correct_label = d.split('\t')[0].split(',')
            if len(correct_label) == 1 and correct_label[0] != '0':
                single_index.append(str(i))
        self.single_result = [','.join(single_index)]

    def save_single_result(self, file_name):
        with open(file_name, 'w')as f:
            [f.write(r + '\n') for r in self.single_result]
        return

    def label(self, label_list):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, d in zip(label_data, self.correct_data):
            correct = [int(num) - 1 for num in d.split('\t')[0].split(',')]
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
        m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count, s_result

    def label_init(self, label_list, init_threshold=0.7):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, d in zip(label_data, self.correct_data):
            correct = [int(num) - 1 for num in d.split('\t')[0].split(',')]
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
        m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count, s_result

    def label_mix_align(self, label_list, align_list, weight=0.5):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, d, align in zip(label_data, self.correct_data, align_list):
            label = weight * label + (1 - weight) * align
            correct = [int(num) - 1 for num in d.split('\t')[0].split(',')]
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
        m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count, s_result

    def label_mix_aligh_init(self, label_list, align_list, init_threshold=0.7, weight=0.5):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, d, align in zip(label_data, self.correct_data, align_list):
            label = weight * label + (1 - weight) * align
            # label = weight + align
            correct = [int(num) - 1 for num in d.split('\t')[0].split(',')]
            rank = []
            # threshold = init_threshold * weight * (1 / len(label))
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
        m_rate, m_count = self.multiple(rank_list)

        return s_rate, s_count, m_rate, m_count, s_result

    def single(self, rank_list, method):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        result = [method]
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
            if count_num == 1:
                score_dic[sent_num][1] += 1
                if correct:
                    score_dic[sent_num][0] += 1
                    result.append('1')
                else:
                    result.append('0')

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

        s_rate, _, _, _, s_result = self.label(label_list)
        key = 'normal'
        best_param_dic[key] = s_rate[-1]
        self.single_result.append(s_result)
        # print('{}\t{}'.format(key, ' '.join(s_rate)))

        for i in range(1, 10 + 1, 2):
            init_threshold = round(i * 0.1, 1)
            s_rate, _, _, _, s_result = self.label_init(label_list, init_threshold)
            key = 'init {}'.format(init_threshold)
            best_param_dic[key] = s_rate[-1]
            self.single_result.append(s_result)
            # print('{}\t{}'.format(key, ' '.join(s_rate)))

        if align_list:
            for i in range(1, 10 + 1, 2):
                weight = round(i * 0.1, 1)
                s_rate, _, _, _, s_result = self.label_mix_align(label_list, align_list, weight)
                key = 'mix {}'.format(weight)
                best_param_dic[key] = s_rate[-1]
                self.single_result.append(s_result)
                # print('{}\t{}'.format(key, ' '.join(s_rate)))

            for i in range(1, 10 + 1, 2):
                weight = round(i * 0.1, 1)
                for j in range(1, 10 + 1, 2):
                    init_threshold = round(j * 0.1, 1)
                    s_rate, _, _, _, s_result = self.label_mix_aligh_init(label_list, align_list, init_threshold, weight)
                    key = 'init {} mix {}'.format(init_threshold, weight)
                    best_param_dic[key] = s_rate[-1]
                    self.single_result.append(s_result)
                    # print('{}\t{}'.format(key, ' '.join(s_rate)))

        return best_param_dic


def model_type(t):
    if t == 'l':
        return 'Local'
    elif t == 'lr':
        return 'Local_Reg'
    elif t == 's':
        return 'Server'
    else:
        return 'Server_Reg'


if __name__ == '__main__':
    args = sys.argv
    model_name = args[1][:-4]
    model_dir = re.search(r'^(.*/)', args[1]).group(1)

    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    config_file = config_files[0]
    config = configparser.ConfigParser()
    config.read(config_file)

    data_type = model_dir.split('_')[2]
    section = model_type(data_type)
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

    evaluater = Evaluate(correct)
    best_param_dic = evaluater.param_search(label, align)
    evaluater.save_single_result(args[1][:-4] + '.E.s_res.csv')

    print('Best parm')
    for k, v in sorted(best_param_dic.items(), key=lambda x: x[1], reverse=True)[:1]:
        print('{} {}'.format(k, v))
        k = k.split(' ')
        if len(k) == 1:
            s_rate, s_count, m_rate, m_count, _ = evaluater.label(label)
        elif len(k) == 2:
            if k[0] == 'init':
                s_rate, s_count, m_rate, m_count, _ = evaluater.label_init(label, float(k[1]))
            else:
                s_rate, s_count, m_rate, m_count, _ = evaluater.label_mix_align(label, align, float(k[1]))
        else:
            s_rate, s_count, m_rate, m_count, _ = evaluater.label_mix_aligh_init(label, align, float(k[1]), float(k[3]))

        print('s: {} | {}'.format(' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
        print('m: {} | {}'.format(' '.join(x for x in m_rate), ' '.join(x for x in m_count)))
