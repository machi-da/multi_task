import argparse
import re
import os
import glob
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


def merge(output_dir, model_dir1, model_dir2, correct_label, correct_index, valid_num):
    result_dic = OrderedDict()
    model_file_num = len(glob.glob(os.path.join(model_dir1, 'model_epoch_*.npz')))

    for i in tqdm(range(1, model_file_num + 1)):
        for j in range(1, model_file_num + 1):
            label, _, _, _ = evaluate.load_score_file(model_dir1 + 'model_epoch_{}'.format(i), model_dir1)
            _, align, _, _ = evaluate.load_score_file(model_dir2 + 'model_epoch_{}'.format(j), model_dir2)
            model = 'label{}_encdec{}'.format(i, j)

            try:
                s_total, s_result = gridsearch.main(label, align, correct_label, correct_index, valid_num, align_only=False, print_flag=False)
                result_dic[model] = s_total
                with open(output_dir + model + '.s_res.csv', 'w')as f:
                    [f.write('{}\t{}\n'.format(l[0], l[1])) for l in sorted(s_result, key=lambda x: x[0])]
            except KeyError:
                result_dic[model] = 0
            except ValueError:
                result_dic[model] = -1

    with open(output_dir + 'merge_result.txt', 'w')as f:
        [f.write('{} {}\t{}\n'.format(i, k, v)) for i, (k, v) in enumerate(result_dic.items(), start=1)]
        best_comb = max(result_dic, key=(lambda x: result_dic[x]))
        f.write('[Best comb]\n')
        f.write('{}\t{}\n'.format(best_comb, result_dic[best_comb]))
    return result_dic[best_comb]


def main():
    args = parse_args()
    model_name1 = args.label_model
    model_dir1 = re.search(r'^(.*/)', model_name1).group(1)

    model_name2 = args.encdec_model
    model_dir2 = re.search(r'^(.*/)', model_name2).group(1)

    valid_num = args.valid_num

    valid_file_num = len(glob.glob(os.path.join(model_dir1, 'valid*/')))
    if valid_file_num:
        _, _, correct_label, correct_index = evaluate.load_score_file('', model_dir1)
        slice_size = len(correct_label) // 5
        correct_label, correct_index = gridsearch.shuffle_list(correct_label, correct_index)
        correct_label = gridsearch.slice_list(correct_label, slice_size)
        correct_index = gridsearch.slice_list(correct_index, slice_size)

        for i in range(1, valid_file_num + 1):
            m1 = model_dir1 + 'valid{}/'.format(i)
            m2 = model_dir2 + 'valid{}/'.format(i)
            output_dir = m1 + model_dir2
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            merge(output_dir, m1, m2, correct_label[i-1], correct_index[i-1], valid_num)

    else:
        label, _, correct_label, correct_index = evaluate.load_score_file(model_dir1, model_name1)
        _, align, _, _ = evaluate.load_score_file(model_dir2, model_name2)
        output_dir = model_name1 + model_dir2
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        merge(output_dir, model_dir1, model_dir2, correct_label, correct_index, valid_num)


if __name__ == '__main__':
    main()