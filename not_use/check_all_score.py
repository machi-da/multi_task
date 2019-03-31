import argparse
from old import evaluate, gridsearch


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
        result_dic[i] = s_rate
    k = max(result_dic, key=lambda x: result_dic[x])
    print(k, result_dic[k])

    # correct_label_data, _, _ = dataset.load_with_label_index('/home/lr/machida/yahoo/evaluate/correct1-2.txt.single.split')
    # slice_size = len(correct_label_data) // 5
    # correct_label_data, = gridsearch.shuffle_list(correct_label_data)
    # split_correct_label = gridsearch.slice_list(correct_label_data, slice_size)
    # ev = evaluate.Evaluate()

    # for ite in range(1, 5 + 1):
    #     for i in range(1, 10 + 1):
    #         model_name = model_dir + 'valid{}/model_epoch_{}'.format(ite, i)
    #         label, align, correct_label, single_index = evaluate.load_score_file(model_name, model_dir)
    #         c_train, c_dev, c_test = gridsearch.split_train_dev_test(split_correct_label, ite - 1)
    #         print(i)
    #         ev.correct_label = c_test
    #         best_param_dic = ev.param_search(align, [])
    #         k = max(best_param_dic, key=lambda x: best_param_dic[x])
    #         v = best_param_dic[k]
    #         result_dic[i] = s_rate
    #     k = max(result_dic, key=lambda x: result_dic[x])
    #     print(k, result_dic[k])


if __name__ == '__main__':
    main()