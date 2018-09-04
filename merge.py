import argparse
import re

import evaluate
import gridsearch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('label_model')
    parser.add_argument('encdec_model')
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
    label, _, correct = evaluate.load_score_file(model_name1, model_dir1)

    model_name2 = args.encdec_model
    model_dir2 = re.search(r'^(.*/)', model_name2).group(1)
    _, align, _ = evaluate.load_score_file(model_name2, model_dir2)
    model_name2 = model_name2.replace('/model_epoch', '')

    gridsearch.main(model_name1 + '.merge.' + model_name2, label, align, correct)


if __name__ == '__main__':
    main()