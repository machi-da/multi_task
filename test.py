import argparse
import configparser
import os
import re
import glob
import logging
import numpy as np
from logging import getLogger
import dataset
import convert

import gridsearch
import model_reg

np.set_printoptions(precision=3)
os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_file = args.model_file
    model_dir = re.search(r'(.*/)', model_file).group(1)
    """LOAD CONFIG FILE"""
    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    assert len(config_files) == 1, 'Put only one config file in the directory'
    config_file = config_files[0]
    config = configparser.ConfigParser()
    config.read(config_file)
    """LOGGER"""
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file = model_dir + 'test_log.txt'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('[Test start] logging to {}'.format(log_file))
    """PARAMATER"""
    embed_size = int(config['Parameter']['embed_size'])
    hidden_size = int(config['Parameter']['hidden_size'])
    class_size = int(config['Parameter']['class_size'])
    dropout_ratio = float(config['Parameter']['dropout'])
    coefficient = float(config['Parameter']['coefficient'])
    """TEST DETAIL"""
    gpu_id = args.gpu
    batch_size = args.batch
    data = model_dir.split('/')[-2].split('_')
    
    model_type = data[0]
    if 'normal' in data[1]:
        vocab_type = 'normal'
    else:
        vocab_type = 'subword'
    if data[2] == 's':
        data_path = 'server'
    else:
        data_path = 'local'

    """DATASET"""
    test_src_file = config[data_path]['test_src_file']
    row_score_file = config[data_path]['row_score_file']
    row_score = dataset.txt_to_list(row_score_file)

    test_data_size = dataset.data_size(test_src_file)
    logger.info('test size: {}'.format(test_data_size))
    if vocab_type == 'normal':
        src_vocab = dataset.VocabNormal()
        src_vocab.load(model_dir + 'src_vocab.normal.pkl')
        src_vocab.set_reverse_vocab()
        trg_vocab = dataset.VocabNormal()
        trg_vocab.load(model_dir + 'trg_vocab.normal.pkl')
        trg_vocab.set_reverse_vocab()

        sos = convert.convert_list(np.array([src_vocab.vocab['<s>']], dtype=np.int32), gpu_id)
        eos = convert.convert_list(np.array([src_vocab.vocab['</s>']], dtype=np.int32), gpu_id)

    elif vocab_type == 'subword':
        src_vocab = dataset.VocabSubword()
        src_vocab.load(model_dir + 'src_vocab.sub.model')
        trg_vocab = dataset.VocabSubword()
        trg_vocab.load(model_dir + 'trg_vocab.sub.model')

        sos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('<s>')], dtype=np.int32), gpu_id)
        eos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('</s>')], dtype=np.int32), gpu_id)

    src_vocab_size = len(src_vocab.vocab)
    trg_vocab_size = len(trg_vocab.vocab)
    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    test_iter = dataset.Iterator(test_src_file, test_src_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False)

    gridsearcher = gridsearch.GridSearch(test_src_file)
    """MODEL"""
    if model_type == 'multi':
        model = model_reg.Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient)
    elif model_type in ['label', 'pretrain']:
        model = model_reg.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio)
    else:
        model = model_reg.EncoderDecoder(src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_ratio)

    chainer.serializers.load_npz(model_file, model)
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TEST"""
    epoch = 'T'
    outputs = []
    labels = []
    alignments = []
    for i, batch in enumerate(test_iter.generate(), start=1):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            output, label, align = model.predict(batch[0], sos, eos)
        for o in output:
            outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
        for l in label:
            labels.append(chainer.cuda.to_cpu(l))
        for a in align:
            alignments.append(chainer.cuda.to_cpu(a))

    model_file = model_file[:-3]
    if model_type == 'multi':
        score = gridsearcher.split_data(labels, alignments)
        logger.info('E{} ## {}'.format(epoch, score[0]))
        logger.info('E{} ## {}'.format(epoch, score[1]))
        with open(model_file + 'label.T', 'w')as f:
            [f.write('{}\n'.format(l)) for l in labels]
        with open(model_file + '.hypo.T', 'w')as f:
            [f.write(o + '\n') for o in outputs]
        with open(model_file + '.align.T', 'w')as f:
            [f.write('{}\n'.format(a)) for a in alignments]

    elif model_type in ['label', 'pretrain']:
        score = gridsearcher.split_data(labels, alignments)
        logger.info('E{} ## {}'.format(epoch, score[0]))
        logger.info('E{} ## {}'.format(epoch, score[1]))
        with open(model_file + 'label.T', 'w')as f:
            [f.write('{}\n'.format(l)) for l in labels]

    else:
        score = gridsearcher.split_data(row_score, alignments)
        logger.info('E{} ## {}'.format(epoch, score[0]))
        logger.info('E{} ## {}'.format(epoch, score[1]))
        with open(model_file + '.hypo.T', 'w')as f:
            [f.write(o + '\n') for o in outputs]
        with open(model_file + '.align.T', 'w')as f:
            [f.write('{}\n'.format(a)) for a in alignments]


if __name__ == '__main__':
    main()