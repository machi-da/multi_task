import argparse
import re
import os
from collections import OrderedDict
from tqdm import tqdm

import evaluate
import gridsearch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_model')
    parser.add_argument('encdec_model')
    parser.add_argument('--valid_num', '-v', type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name1 = args.label_model
    model_dir1 = re.search(r'^(.*/)', model_name1).group(1)

    model_name2 = args.encdec_model
    model_dir2 = re.search(r'^(.*?/)', model_name2).group(1)

    valid_num = args.valid_num

    merge_dir = model_dir1 + model_dir2
    if not os.path.exists(merge_dir):
        os.mkdir(merge_dir)

    max_epoch_num = 10
    result_dic = OrderedDict()
    for i in tqdm(range(1, max_epoch_num + 1)):
        for j in range(1, max_epoch_num + 1):
            label, _, correct_label, correct_index = evaluate.load_score_file(model_dir1 + 'model_epoch_{}'.format(i), model_dir1)
            _, align, _, _ = evaluate.load_score_file(model_dir2 + 'model_epoch_{}'.format(j), model_dir2)
            model = 'label{}_encdec{}'.format(i, j)

            try:
                s_total = gridsearch.main(label, align, correct_label, correct_index, valid_num, align_only=False, print_flag=False)
                result_dic[model] = s_total
            except KeyError:
                result_dic[model] = 0
            except ValueError:
                result_dic[model] = -1

    with open(merge_dir + 'merge_result.txt', 'w')as f:
        [f.write('{} {}\t{}\n'.format(i, k, v)) for i, (k, v) in enumerate(result_dic.items(), start=1)]
        best_comb = max(result_dic, key=(lambda x: result_dic[x]))
        f.write('[Best comb]\n')
        f.write('{}\t{}\n'.format(best_comb, result_dic[best_comb]))


if __name__ == '__main__':
    main()