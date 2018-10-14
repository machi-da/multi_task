import argparse
import re
import os

import evaluate
import gridsearch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_model')
    parser.add_argument('encdec_model')
    parser.add_argument('--assign', '-a', action='store_true')
    args = parser.parse_args()
    return args


def model_type(model_dir):
    model_detail = model_dir.split('_')
    if model_detail[0] == 'label':
        return '.label'
    elif model_detail[0] == 'encdec':
        return '.align'


def main():
    args = parse_args()
    model_name1 = args.label_model
    model_dir1 = re.search(r'^(.*/)', model_name1).group(1)

    model_name2 = args.encdec_model
    model_dir2 = re.search(r'^(.*/)', model_name2).group(1)

    assign = args.assign

    merge_dir = model_dir1[:-1] + model_dir2
    if not os.path.exists(merge_dir):
        os.mkdir(merge_dir)

    if assign:
        model_num1 = model_name1.split('_')[-1].replace('.label', '')
        model_num2 = model_name1.split('_')[-1].replace('.align', '')
        label, _, correct = evaluate.load_score_file(model_name1, model_dir1)
        _, align, _ = evaluate.load_score_file(model_name2, model_dir2)

        save_file = merge_dir + 'label{}_encdec{}'.format(model_num1, model_num2)
        gridsearch.main(save_file, label, align, correct)

    else:
        result_dic = {}
        max_epoch_num = 10
        for i in range(1, max_epoch_num + 1):
            for j in range(1, max_epoch_num + 1):
                label, _, correct = evaluate.load_score_file(model_dir1 + 'model_epoch_{}', model_dir1)
                _, align, _ = evaluate.load_score_file(model_dir2 + 'model_epoch_{}', model_dir2)
                save_file = merge_dir + 'label{}_encdec{}'.format(i, j)
                s_rate = gridsearch.main(save_file, label, align, correct)

                result_dic[save_file] = s_rate
        with open(merge_dir + 'merge_result.txt', 'w')as f:
            [f.write('{}\t{}\n'.format(k, v)) for k, v in result_dic.items()]
        best_comb = max(result_dic, key=(lambda x: result_dic[x]))
        f.write('[Best comb]')
        f.write('{}\t{}\n'.format(best_comb, result_dic[best_comb]))


if __name__ == '__main__':
    main()