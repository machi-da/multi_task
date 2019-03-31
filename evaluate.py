import copy
import numpy as np


class Evaluate:
    def __init__(self):
        self.dic = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}

    def label(self, label_data, test_data):
        dic = copy.deepcopy(self.dic)
        tf_lit = []
        for label, d in zip(label_data, test_data):
            sent_num = d['sent_num'] if d['sent_num'] <= 7 else 7
            if label.argmax() == d['label']:
                dic[sent_num][0] += 1
                tf_lit.append('T')
            else:
                tf_lit.append('F')
            dic[sent_num][1] += 1

        rate = [v[0]/v[1] for k, v in dic.items()]
        count = ['{}/{}'.format(v[0], v[1]) for k, v in dic.items()]
        correct = sum([v[0] for k, v in dic.items()])
        total = sum([v[1] for k, v in dic.items()])
        count.append('{}/{}'.format(correct, total))

        macro = sum(rate) / len(rate)
        micro = correct / total

        return rate, count, tf_lit, macro, micro

    def label_init(self, label_data, test_data, init_threshold=0.7):
        dic = copy.deepcopy(self.dic)
        tf_lit = []
        for label, d in zip(label_data, test_data):
            sent_num = d['sent_num'] if d['sent_num'] <= 7 else 7
            filter_index = list(np.where(label >= init_threshold)[0])

            if len(filter_index) > 0:
                if filter_index[0] == d['label']:
                    dic[sent_num][0] += 1
                    tf_lit.append('T')
                else:
                    tf_lit.append('F')
                dic[sent_num][1] += 1

            else:
                if label.argmax() == d['label']:
                    dic[sent_num][0] += 1
                    tf_lit.append('T')
                else:
                    tf_lit.append('F')
                dic[sent_num][1] += 1

        rate = [v[0] / v[1] for k, v in dic.items()]
        count = ['{}/{}'.format(v[0], v[1]) for k, v in dic.items()]
        correct = sum([v[0] for k, v in dic.items()])
        total = sum([v[1] for k, v in dic.items()])
        count.append('{}/{}'.format(correct, total))

        macro = sum(rate) / len(rate)
        micro = correct / total

        return rate, count, tf_lit, macro, micro

    def label_mix_align(self, label_data, align_data, test_data, weight=0.5):
        dic = copy.deepcopy(self.dic)
        tf_lit = []
        for label, align, d in zip(label_data, align_data, test_data):
            sent_num = d['sent_num'] if d['sent_num'] <= 7 else 7
            label = weight * label + (1 - weight) * align

            if label.argmax() == d['label']:
                dic[sent_num][0] += 1
                tf_lit.append('T')
            else:
                tf_lit.append('F')
            dic[sent_num][1] += 1

        rate = [v[0] / v[1] for k, v in dic.items()]
        count = ['{}/{}'.format(v[0], v[1]) for k, v in dic.items()]
        correct = sum([v[0] for k, v in dic.items()])
        total = sum([v[1] for k, v in dic.items()])
        count.append('{}/{}'.format(correct, total))

        macro = sum(rate) / len(rate)
        micro = correct / total

        return rate, count, tf_lit, macro, micro


    def label_mix_aligh_init(self, label_data, align_data, test_data, init_threshold=0.7, weight=0.5):
        dic = copy.deepcopy(self.dic)
        tf_lit = []
        for label, align, d in zip(label_data, align_data, test_data):
            sent_num = d['sent_num'] if d['sent_num'] <= 7 else 7
            label = weight * label + (1 - weight) * align
            filter_index = list(np.where(label >= init_threshold)[0])

            if len(filter_index) > 0:
                if filter_index[0] == d['label']:
                    dic[sent_num][0] += 1
                    tf_lit.append('T')
                else:
                    tf_lit.append('F')
                dic[sent_num][1] += 1

            else:
                if label.argmax() == d['label']:
                    dic[sent_num][0] += 1
                    tf_lit.append('T')
                else:
                    tf_lit.append('F')
                dic[sent_num][1] += 1

        rate = [v[0] / v[1] for k, v in dic.items()]
        count = ['{}/{}'.format(v[0], v[1]) for k, v in dic.items()]
        correct = sum([v[0] for k, v in dic.items()])
        total = sum([v[1] for k, v in dic.items()])
        count.append('{}/{}'.format(correct, total))

        macro = sum(rate) / len(rate)
        micro = correct / total

        return rate, count, tf_lit, macro, micro

    def param_search(self, label_data, align_data, test_data):
        best_param_dic = {}

        _, _, _, macro, micro = self.label(label_data, test_data)
        key = 'normal'
        best_param_dic[key] = {'macro': macro, 'micro': micro}

        # range: 1~9
        # for i in range(1, 10):
        #     init_threshold = round(i * 0.1, 1)
        #     _, _, _, macro, micro = self.label_init(label_data, test_data, init_threshold)
        #     key = 'init {}'.format(init_threshold)
        #     best_param_dic[key] = {'macro': macro, 'micro': micro}

        if align_data:
            for i in range(1, 10):
                weight = round(i * 0.1, 1)
                _, _, _, macro, micro = self.label_mix_align(label_data, align_data, test_data, weight)
                key = 'mix {}'.format(weight)
                best_param_dic[key] = {'macro': macro, 'micro': micro}

            # for i in range(1, 10):
            #     weight = round(i * 0.1, 1)
            #     for j in range(1, 10):
            #         init_threshold = round(j * 0.1, 1)
            #         _, _, _, macro, micro = self.label_mix_aligh_init(label_data, align_data, test_data, init_threshold, weight)
            #         key = 'init {} mix {}'.format(init_threshold, weight)
            #         best_param_dic[key] = {'macro': macro, 'micro': micro}

        return best_param_dic

    def eval_param(self, label_data, align_data, test_data, init=-1, mix=-1):
        if init == -1:
            if mix == -1:
                rate, count, tf_lit, macro, micro = self.label(label_data, test_data)
            else:
                rate, count, tf_lit, macro, micro = self.label_mix_align(label_data, align_data, test_data, mix)
        else:
            if mix == -1:
                rate, count, tf_lit, macro, micro = self.label_init(label_data, test_data, init)
            else:
                rate, count, tf_lit, macro, micro = self.label_mix_aligh_init(label_data, align_data, test_data, init, mix)

        return rate, count, tf_lit, macro, micro


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