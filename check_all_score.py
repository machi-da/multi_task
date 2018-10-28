import argparse
import evaluate
import gridsearch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--align_only', '-a', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_dir = args.model_dir
    align_only = args.align_only

    result_dic = {}
    for i in range(1, 10 + 1):
        model_name = model_dir + 'model_epoch_{}'.format(i)
        label, align, correct_label, single_index = evaluate.load_score_file(model_name, model_dir)
        print(i)
        s_rate = gridsearch.main(model_name, label, align, correct_label, single_index, detail_flag=False, align_only=align_only)
        result_dic[i] = float(s_rate[-1])
    k = max(result_dic, key=lambda x: result_dic[x])
    print(k, result_dic[k])


if __name__ == '__main__':
    main()