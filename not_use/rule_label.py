import sys
import re
from tqdm import tqdm


class Rulebase:
    def __init__(self, file_name):
        self.pattern = re.compile(r'\?|？|を教え|か。|サイト|方法|を探し|って|知る|知って|知り')
        # self.pattern = re.compile(r'\?|？')

        with open(file_name, 'r')as f:
            self.data = f.readlines()

    def rule(self, sentence):
        sentence = sentence.replace(' ', '')
        if re.search(self.pattern, sentence):
            return True
        else:
            return False

    def check(self, sentences):
        for i, sentence in enumerate(sentences, start=1):
            if self.rule(sentence):
                return i
        return 1

    def main(self, file_name):
        res = []
        for d in tqdm(self.data):
            label, sentences = d.strip().split('\t')
            split_sentences = sentences.split('|||')

            new_label = self.check(split_sentences)

            res.append('{}\t{}\n'.format(new_label, sentences))

        with open(file_name + '.rule', 'w')as f:
            [f.write(r) for r in res]

    def rule_score(self):
        lead_dic = {i: [0, 0] for i in range(2, 7 + 1)}
        rule_dic = {i: [0, 0] for i in range(2, 7 + 1)}
        predict_lit = []
        lead_tf_lit = []
        for d in tqdm(self.data):
            label, sentences = d.strip().split('\t')
            label = int(label)
            sentences = sentences.split('|||')

            sent_num = len(sentences) if len(sentences) <= 7 else 7
            predict = self.check(sentences)
            predict_lit.append(predict)

            if label == 1:
                lead_dic[sent_num][0] += 1
                lead_tf_lit.append('T')
            else:
                lead_tf_lit.append('F')
            lead_dic[sent_num][1] += 1

            if label == predict:
                rule_dic[sent_num][0] += 1
            rule_dic[sent_num][1] += 1

        res = [v[0] / v[1] for k, v in lead_dic.items()]
        total = round(sum(res) / len(res), 3)
        res = [str(round(r, 3)) for r in res]
        print('lead: {} {}'.format(' '.join(res), total))

        res = [v[0] / v[1] for k, v in rule_dic.items()]
        total = round(sum(res) / len(res), 3)
        res = [str(round(r, 3)) for r in res]
        print('rule: {} {}'.format(' '.join(res), total))

        with open('lead_tf.txt', 'w')as f:
            [f.write(l + '\n') for l in lead_tf_lit]
        print('Write lead_tf.txt')


def main():
    args = sys.argv
    file_name = args[1]

    rul = Rulebase(file_name)
    # rul.main()
    rul.rule_score()

if __name__ == '__main__':
    main()