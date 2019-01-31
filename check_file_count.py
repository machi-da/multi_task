import sys
from collections import defaultdict
"""
文数と単語数を数える
"""


def main():
    args = sys.argv
    file_name = args[1]
    file_type = args[2]

    with open(file_name, 'r')as f:
        data = f.readlines()

    sent_dic = defaultdict(int)
    word_dic = defaultdict(int)

    if file_type == 'que':
        for d in data:
            sentences = d.strip().split('\t')[1].split('|||')
            sent_dic[len(sentences)] += 1
            for sentence in sentences:
                words = sentence.split(' ')
                word_dic[len(words)] += 1

        print(sent_dic)
        print(word_dic)

    elif file_type == 'ans':
        for d in data:
            words = d.strip().split(' ')
            word_dic[len(words)] += 1

        print(word_dic)


if __name__ == '__main__':
    main()