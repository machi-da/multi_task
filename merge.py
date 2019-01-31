"""
separate learningモデル用
labelとencdecを組み合わせてスコアを出す
"""
import argparse
import configparser
import sys
import re
import os
import glob
from tqdm import tqdm

import dataset_new
import evaluate_new
import gridsearch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_model')
    parser.add_argument('encdec_model')
    parser.add_argument('--valid', '-v', action='store_true')
    args = parser.parse_args()
    return args


def main():
    """
    model1: label
    model2: encdec を指定する
    """
    args = parse_args()
    model_name1 = args.label_model
    model_dir1 = re.search(r'^(.*/)', model_name1).group(1)

    model_name2 = args.encdec_model
    model_dir2 = re.search(r'^(.*/)', model_name2).group(1)

    valid = args.valid

    # 結果保存用ディレクトリ作成
    output_dir = model_dir1 + model_dir2
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 評価データ準備
    config = configparser.ConfigParser()
    config_files = glob.glob(os.path.join(model_dir1, '*.ini'))
    config.read(config_files[0])
    valid_num = int(config['Parameter']['valid_num'])
    test_src_file = config['server']['single_src_file']
    test_trg_file = config['server']['single_trg_file']
    data = dataset_new.load_label_corpus_file(test_src_file, test_trg_file)
    data_sub_lit = dataset_new.split_valid_data(data, valid_num)

    evaluater = evaluate_new.Evaluate()

    result_dic = {}
    # validファイルに分割されている時
    if valid:
        model_file_num = len(glob.glob(os.path.join(model_dir1, 'valid1/model_epoch_*.npz')))

        label_dic = {}
        align_dic = {}
        for i in range(1, model_file_num + 1):
            label_dic[i] = []
            align_dic[i] = []
            for valid in [2, 3, 4, 5, 1]:
                label, _ = dataset_new.load_score_file(model_dir1 + 'valid{}/model_epoch_{}'.format(valid, i))
                label_dic[i].append(label)
                _, align = dataset_new.load_score_file(model_dir2 + 'valid{}/model_epoch_{}'.format(valid, i))
                align_dic[i].append(align)

        order = {1: [4, 5], 2: [5, 1], 3: [1, 2], 4: [2, 3], 5: [3, 4]}

        for i in tqdm(range(1, model_file_num + 1)):
            for j in range(1, model_file_num + 1):
                info = []
                for ite, v in order.items():
                    _, dev_data, test_data = dataset_new.separate_train_dev_test(data_sub_lit, ite)
                    dev_label = label_dic[i][v[0] - 1]
                    test_label = label_dic[i][v[1] - 1]

                    dev_align = align_dic[j][v[0] - 1]
                    test_align = align_dic[j][v[1] - 1]

                    best_param_dic = evaluater.param_search(dev_label, dev_align, dev_data)
                    param = max(best_param_dic, key=lambda x: best_param_dic[x]['macro'])
                    init, mix = evaluate_new.key_to_param(param)
                    dev_score = round(best_param_dic[param]['macro'], 3)

                    rate, count, tf_lit, macro, micro = evaluater.eval_param(test_label, test_align, test_data, init, mix)
                    test_macro_score = round(macro, 3)
                    test_micro_score = round(micro, 3)
                    info.append({
                        'dev_score': dev_score,
                        'param': param,
                        'macro': test_macro_score,
                        'micro': test_micro_score
                    })

                ave_dev_score, ave_macro_score, ave_micro_score = 0, 0, 0
                param = []

                for v, r in enumerate(info, start=1):
                    ave_dev_score += r['dev_score']
                    ave_macro_score += r['macro']
                    ave_micro_score += r['micro']
                    param.append(r['param'])

                ave_dev_score = round(ave_dev_score / valid_num, 3)
                ave_macro_score = round(ave_macro_score / valid_num, 3)
                ave_micro_score = round(ave_micro_score / valid_num, 3)

                key = 'label{}_enc{}'.format(i, j)
                result_dic[key] = {
                    'dev': ave_dev_score,
                    'micro': ave_micro_score,
                    'macro': ave_macro_score,
                    'param': ' '.join(param)
                }

        best_score = max(result_dic, key=lambda x: result_dic[x]['dev'])
        with open(output_dir + 'merge.txt', 'w')as f:
            [f.write('{}: {}\n'.format(k, v)) for k, v in sorted(result_dic.items())]
            f.write('best score\n{}: {}\n'.format(best_score, result_dic[best_score]))

    else:
        model_file_num = len(glob.glob(os.path.join(model_dir1, 'model_epoch_*.npz')))
        for i in tqdm(range(1, model_file_num + 1)):
            label, _ = dataset_new.load_score_file(model_dir1 + 'model_epoch_{}'.format(i))
            label_sub_lit = dataset_new.split_valid_data(label, valid_num)
            for j in range(1, model_file_num + 1):
                _, align = dataset_new.load_score_file(model_dir2 + 'model_epoch_{}'.format(j))
                align_sub_lit = dataset_new.split_valid_data(align, valid_num)
                info = []
                for ite in range(1, valid_num + 1):
                    _, dev_data, test_data = dataset_new.separate_train_dev_test(data_sub_lit, ite)
                    _, dev_label, test_label = dataset_new.separate_train_dev_test(label_sub_lit, ite)
                    _, dev_align, test_align = dataset_new.separate_train_dev_test(align_sub_lit, ite)

                    best_param_dic = evaluater.param_search(dev_label, dev_align, dev_data)
                    param = max(best_param_dic, key=lambda x: best_param_dic[x]['macro'])
                    init, mix = evaluate_new.key_to_param(param)
                    dev_score = round(best_param_dic[param]['macro'], 3)

                    rate, count, tf_lit, macro, micro = evaluater.eval_param(test_label, test_align, test_data, init, mix)
                    test_macro_score = round(macro, 3)
                    test_micro_score = round(micro, 3)
                    info.append({
                        'dev_score': dev_score,
                        'param': param,
                        'macro': test_macro_score,
                        'micro': test_micro_score
                    })

                ave_dev_score, ave_macro_score, ave_micro_score = 0, 0, 0
                param = []

                for v, r in enumerate(info, start=1):
                    ave_dev_score += r['dev_score']
                    ave_macro_score += r['macro']
                    ave_micro_score += r['micro']
                    param.append(r['param'])

                ave_dev_score = round(ave_dev_score / valid_num, 3)
                ave_macro_score = round(ave_macro_score / valid_num, 3)
                ave_micro_score = round(ave_micro_score / valid_num, 3)

                key = 'label{}_enc{}'.format(i, j)
                result_dic[key] = {
                    'dev': ave_dev_score,
                    'micro': ave_micro_score,
                    'macro': ave_macro_score,
                    'param': ' '.join(param)
                }

        best_score = max(result_dic, key=lambda x: result_dic[x]['dev'])
        with open(output_dir + 'merge.txt', 'w')as f:
            [f.write('{}: {}\n'.format(k, v)) for k, v in sorted(result_dic.items())]
            f.write('best score\n{}: {}\n'.format(best_score, result_dic[best_score]))



def merge_valid_file():
    args = sys.argv
    model_dir = args[1]

    valid_index = [5, 1, 2, 3, 4]
    model_file_num = len(glob.glob(os.path.join(model_dir, 'valid1/model_epoch_*.npz')))

    config = configparser.ConfigParser()
    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    config.read(config_files[0])
    test_src_file = config['server']['single_src_file']
    correct_label, _, _, correct_index = dataset.load_binary_score_file(test_src_file)
    correct_label, correct_index = gridsearch.shuffle_list(correct_label, correct_index)

    label_dic = {}
    align_dic = {}

    # 初期化
    for i in range(1, model_file_num + 1):
        label_dic[i] = []
        align_dic[i] = []

    # validに分割されたファイルをモデルごとに統合
    for i in valid_index:
        for j in range(1, model_file_num + 1):
            label, _ = evaluate.load_score_file(model_dir + 'valid{}/model_epoch_{}'.format(i, j))
            _, align = evaluate.load_score_file(model_dir + 'valid{}/model_epoch_{}'.format(i, j))

            label_dic[j].extend(label)
            align_dic[j].extend(align)

    # データをソート
    for i in range(1, model_file_num + 1):
        # zip_list = list(zip(correct_index, label_dic[i], align_dic[i]))
        zip_list = list(zip(correct_index, label_dic[i]))
        zip_list.sort()
        # _, l, a = zip(*zip_list)
        _, l = zip(*zip_list)
        label_dic[i] = list(l)
        # align_dic[i] = list(a)

    # 書き出し
    for i in range(1, model_file_num + 1):
        with open(model_dir + 'model_epoch_{}.label'.format(i), 'w')as f:
            [f.write('{}\n'.format(l)) for l in label_dic[i]]
        # with open(model_dir + 'model_epoch_{}.align'.format(i), 'w')as f:
        #     [f.write('{}\n'.format(a)) for a in align_dic[i]]


def a():
    args = parse_args()
    model_name1 = args.label_model
    model_dir1 = re.search(r'^(.*/)', model_name1).group(1)

    model_name2 = args.encdec_model
    model_dir2 = re.search(r'^(.*/)', model_name2).group(1)

    valid = args.valid

    # 結果保存用ディレクトリ作成
    output_dir = model_dir1 + model_dir2
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 評価データ準備
    config = configparser.ConfigParser()
    config_files = glob.glob(os.path.join(model_dir1, '*.ini'))
    config.read(config_files[0])
    valid_num = int(config['Parameter']['valid_num'])
    test_src_file = config['server']['single_src_file']
    test_trg_file = config['server']['single_trg_file']
    data = dataset_new.load_label_corpus_file(test_src_file, test_trg_file)
    data_sub_lit = dataset_new.split_valid_data(data, valid_num)

    evaluater = evaluate_new.Evaluate()

    result_dic = {}

    model_file_num = len(glob.glob(os.path.join(model_dir1, 'valid1/model_epoch_*.npz')))

    label_dic = {}
    for i in range(1, model_file_num + 1):
        label_dic[i] = []
        for valid in [2, 3, 4, 5, 1]:
            label, _ = dataset_new.load_score_file(model_dir1 + 'valid{}/model_epoch_{}'.format(valid, i))
            label_dic[i].append(label)

    for j in range(1, model_file_num + 1):
        _, align = dataset_new.load_score_file(model_dir2 + 'model_epoch_{}'.format(j))
        align_sub_lit = dataset_new.split_valid_data(align, valid_num)

    order = {1: [4, 5], 2: [5, 1], 3: [1, 2], 4: [2, 3], 5: [3, 4]}

    for i in tqdm(range(1, model_file_num + 1)):
        for j in range(1, model_file_num + 1):
            info = []
            for ite, v in order.items():
                _, dev_data, test_data = dataset_new.separate_train_dev_test(data_sub_lit, ite)
                dev_label = label_dic[i][v[0] - 1]
                test_label = label_dic[i][v[1] - 1]

                _, dev_align, test_align = dataset_new.separate_train_dev_test(align_sub_lit, ite)

                best_param_dic = evaluater.param_search(dev_label, dev_align, dev_data)
                param = max(best_param_dic, key=lambda x: best_param_dic[x]['macro'])
                init, mix = evaluate_new.key_to_param(param)
                dev_score = round(best_param_dic[param]['macro'], 3)

                rate, count, tf_lit, macro, micro = evaluater.eval_param(test_label, test_align, test_data, init, mix)
                test_macro_score = round(macro, 3)
                test_micro_score = round(micro, 3)
                info.append({
                    'dev_score': dev_score,
                    'param': param,
                    'macro': test_macro_score,
                    'micro': test_micro_score
                })

            ave_dev_score, ave_macro_score, ave_micro_score = 0, 0, 0
            param = []

            for v, r in enumerate(info, start=1):
                ave_dev_score += r['dev_score']
                ave_macro_score += r['macro']
                ave_micro_score += r['micro']
                param.append(r['param'])

            ave_dev_score = round(ave_dev_score / valid_num, 3)
            ave_macro_score = round(ave_macro_score / valid_num, 3)
            ave_micro_score = round(ave_micro_score / valid_num, 3)

            key = 'label{}_enc{}'.format(i, j)
            result_dic[key] = {
                'dev': ave_dev_score,
                'micro': ave_micro_score,
                'macro': ave_macro_score,
                'param': ' '.join(param)
            }

    best_score = max(result_dic, key=lambda x: result_dic[x]['dev'])
    with open(output_dir + 'merge.txt', 'w')as f:
        [f.write('{}: {}\n'.format(k, v)) for k, v in sorted(result_dic.items())]
        f.write('best score\n{}: {}\n'.format(best_score, result_dic[best_score]))


if __name__ == '__main__':
    # main()
    a()
    # merge_valid_file()