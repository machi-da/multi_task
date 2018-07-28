import sys
import copy
import numpy as np


class Evaluate:
    def __init__(self, correct_txt_file, align_weight=0, label_threshold=0.7):
        with open(correct_txt_file, 'r')as f:
            self.correct_data = f.readlines()
        self.align_weight = align_weight
        self.label_threshold = label_threshold

    def rank(self, label_list, align_list, init_flag=False, align_flag=False):
        label_data = copy.deepcopy(label_list)
        rank_list = []
        for label, d, align in zip(label_data, self.correct_data, align_list):
            if align_flag:
                label = label + (self.align_weight * align)
            correct = [int(num) - 1 for num in d.split('\t')[0].split(',')]
            rank = []
            true_index = list(np.where(label >= self.label_threshold)[0])
            for _ in range(len(label)):
                index = label.argmax()

                # 先頭を優先
                if init_flag:
                    if len(true_index) > 0:
                        if index != true_index[0]:
                            if true_index[0] in label:
                                rank.append((true_index[0], True))
                            else:
                                rank.append((true_index[0], False))
                            label[true_index[0]] = -1
                            true_index = true_index[1:]
                            continue

                if index in correct:
                    rank.append((index, True))
                else:
                    rank.append((index, False))
                label[index] = -1
                if index in true_index:
                    true_index.remove(index)
            rank_list.append(rank)

        return rank_list

    def single(self, rank_list):
        score_dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
        for r in rank_list:
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


if __name__ == '__main__':
    args = sys.argv
    correct = args[1]
    model_name = args[2]

    label = []
    with open(model_name + '.label', 'r')as f:
        label_data = f.readlines()
    for line in label_data:
        line = line[1:-2]
        score = np.array([float(l) for l in line.split()])
        label.append(score)

    align = []
    with open(model_name + '.align', 'r')as f:
        attn_data = f.readlines()
    for line in attn_data:
        line = line[1:-2]
        score = np.array([float(l) for l in line.split()])
        align.append(score)
    
    weight = 1.0
    label_threshold = 0.7
    evaluater = Evaluate(correct, weight, label_threshold)
    
    rank_list = evaluater.rank(label, align)
    s_rate, s_count = evaluater.single(rank_list)
    m_rate, m_count = evaluater.multiple(rank_list)
    print('normal')
    print('s: {} | {}'.format(' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    print('m: {} | {}'.format(' '.join(x for x in m_rate), ' '.join(x for x in m_count)))
    rank_list = evaluater.rank(label, align, init_flag=True)
    s_rate, s_count = evaluater.single(rank_list)
    m_rate, m_count = evaluater.multiple(rank_list)
    print('normal init')
    print('s: {} | {}'.format(' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    print('m: {} | {}'.format(' '.join(x for x in m_rate), ' '.join(x for x in m_count)))
    rank_list = evaluater.rank(label, align, init_flag=True, align_flag=True)
    s_rate, s_count = evaluater.single(rank_list)
    m_rate, m_count = evaluater.multiple(rank_list)
    print('normal init align ')
    print('s: {} | {}'.format(' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    print('m: {} | {}'.format(' '.join(x for x in m_rate), ' '.join(x for x in m_count)))