"""
separate learningモデル用
labelとencdecを組み合わせてスコアを出す
"""
import argparse
import configparser
import re
import os
import glob
from collections import OrderedDict
from tqdm import tqdm

import dataset
import evaluate
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

    valid_index = [5, 1, 2, 3, 4]
    valid_num = len(valid_index)
    gridsearcher = gridsearch.GridSearch(valid_num)

    # 評価データ準備
    config = configparser.ConfigParser()
    config_files = glob.glob(os.path.join(model_dir1, '*.ini'))
    config.read(config_files[0])
    test_src_file = config['server']['single_src_file']
    correct_label, _, _, correct_index = dataset.load_binary_score_file(test_src_file)
    correct_label, correct_index = gridsearch.shuffle_list(correct_label, correct_index)

    accuracy_dic = OrderedDict()
    # validファイルに分割されている時
    if valid:
        model_file_num = len(glob.glob(os.path.join(model_dir1, 'valid1/model_epoch_*.npz')))

        label_dic = {}
        align_dic = {}
        # 初期化
        for i in range(1, model_file_num + 1):
            label_dic[i] = []
            align_dic[i] = []

        # validに分割されたファイルをモデルごとに統合
        for i in valid_index:
            for j in range(1, model_file_num + 1):
                label, _ = evaluate.load_score_file(model_dir1 + 'valid{}/model_epoch_{}'.format(i, j))
                _, align = evaluate.load_score_file(model_dir2 + 'valid{}/model_epoch_{}'.format(i, j))

                label_dic[j].extend(label)
                align_dic[j].extend(align)

        # データをソート
        for i in range(1, model_file_num + 1):
            zip_list = list(zip(correct_index, label_dic[i], align_dic[i]))
            zip_list.sort()
            _, l, a = zip(*zip_list)
            label_dic[i] = list(l)
            align_dic[i] = list(a)

        # スコアの計算
        for i in tqdm(range(1, model_file_num + 1)):
            for j in range(1, model_file_num + 1):
                dev_score, test_score, param_list, test_score_list, s_result_list = gridsearcher.gridsearch(correct_label, correct_index, label_dic[i], align_dic[j])
                key = 'label{}_enc{}'.format(i, j)
                accuracy_dic[key] = [dev_score, test_score, param_list]

                with open(output_dir + key + '.s_res.txt', 'w')as f:
                    [f.write('{}\n'.format(l[1])) for l in sorted(s_result_list, key=lambda x: x[0])]

            with open(output_dir + 'model_epoch_{}.label'.format(i), 'w')as f:
                [f.write('{}\n'.format(l)) for l in label_dic[i]]
            with open(output_dir + 'model_epoch_{}.align'.format(i), 'w')as f:
                [f.write('{}\n'.format(a)) for a in align_dic[i]]

        best_score = max(accuracy_dic, key=lambda x: accuracy_dic[x][0])
        with open(output_dir + 'merge.txt', 'w')as f:
            [f.write('{} {}\n'.format(k, v)) for k, v in accuracy_dic.items()]
            f.write('best score: {}\t{}\n'.format(best_score, accuracy_dic[best_score]))

    else:
        model_file_num = len(glob.glob(os.path.join(model_dir1, 'model_epoch_*.npz')))
        for i in tqdm(range(1, model_file_num + 1)):
            label, _ = evaluate.load_score_file(model_dir1 + 'model_epoch_{}'.format(i))
            for j in range(1, model_file_num + 1):
                _, align = evaluate.load_score_file(model_dir2 + 'model_epoch_{}'.format(j))

                dev_score, test_score, param_list, test_score_list, s_result_list = gridsearcher.gridsearch(correct_label, correct_index, label, align)
                key = 'label{}_enc{}'.format(i, j)
                accuracy_dic[key] = [dev_score, test_score, param_list]
                with open(output_dir + key + '.s_res.txt', 'w')as f:
                    [f.write('{}\n'.format(l[1])) for l in sorted(s_result_list, key=lambda x: x[0])]

        best_score = max(accuracy_dic, key=lambda x: accuracy_dic[x][0])
        with open(output_dir + 'merge.txt', 'w')as f:
            [f.write('{} {}\n'.format(k, v)) for k, v in accuracy_dic.items()]
            f.write('best score: {}\t{}\n'.format(best_score, accuracy_dic[best_score]))


if __name__ == '__main__':
    main()