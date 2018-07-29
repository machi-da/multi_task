import argparse
import configparser
import os
import glob
import logging
import numpy as np
from logging import getLogger
import dataset
import convert

import evaluate
from model import Multi
from model_reg import MultiReg

# os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', type=str, required=True)
    args = parser.parse_args()
    return args


def model_type(t):
    if t == 'l':
        return 'Local'
    elif t == 'lr':
        return 'Local_Reg'
    elif t == 's':
        return 'Server'
    else:
        return 'Server_Reg'


def main():
    args = parse_args()
    model_dir = args.model_dir
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
    align_weight = float(config['Parameter']['align_weight'])
    multi = bool(int(config['Parameter']['multi']))
    """TEST DETAIL"""
    gpu_id = args.gpu
    batch_size = args.batch
    model_file = model_dir + 'model_epoch_{}.npz'.format(args.model)
    data_type = model_dir.split('_')[2]
    reg = False if data_type == 'l' or data_type == 's' else True
    if 'normal' in model_dir:
        vocab_type = 'normal'
    else:
        vocab_type = 'subword'
    """DATASET"""
    section = model_type(data_type)
    test_src_file = config[section]['test_src_file']

    test_data_size = dataset.data_size(test_src_file)
    logger.info('test size: {0}'.format(test_data_size))
    if vocab_type == 'normal':
        src_vocab = dataset.VocabNormal(reg)
        src_vocab.load(model_dir + 'src_vocab.normal.pkl')
        src_vocab.set_reverse_vocab()
        trg_vocab = dataset.VocabNormal(reg)
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

    evaluater = evaluate.Evaluate(test_src_file, align_weight)
    test_iter = dataset.Iterator(test_src_file, test_src_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False)
    """MODEL"""
    if reg:
        class_size = 1
        model = MultiReg(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, multi=multi)
    else:
        model = Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, multi=multi)
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
        batch = convert.convert(batch, gpu_id)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            output, label, align = model.predict(batch[0], sos, eos)
        for l in label:
            labels.append(l)
        if multi:
            for o, a in zip(output, align):
                o = chainer.cuda.to_cpu(o)
                outputs.append(trg_vocab.id2word(o))
                alignments.append(a)
    rank_list = evaluater.rank(labels)
    s_rate, s_count = evaluater.single(rank_list)
    m_rate, m_count = evaluater.multiple(rank_list)
    logger.info('E{} ## normal'.format(epoch))
    logger.info('E{} ## s: {} | {}'.format(epoch, ' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    logger.info('E{} ## m: {} | {}'.format(epoch, ' '.join(x for x in m_rate), ' '.join(x for x in m_count)))
    rank_list = evaluater.rank_init(labels)
    s_rate_i, s_count = evaluater.single(rank_list)
    m_rate_i, m_count = evaluater.multiple(rank_list)
    logger.info('E{} ## normal init'.format(epoch))
    logger.info('E{} ## s: {} | {}'.format(epoch, ' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
    logger.info('E{} ## m: {} | {}'.format(epoch, ' '.join(x for x in m_rate), ' '.join(x for x in m_count)))
    if multi:
        rank_list = evaluater.rank_init_align(labels, alignments)
        s_rate_a, s_count = evaluater.single(rank_list)
        m_rate_a, m_count = evaluater.multiple(rank_list)
        logger.info('E{} ## normal init align'.format(epoch))
        logger.info('E{} ## s: {} | {}'.format(epoch, ' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
        logger.info('E{} ## m: {} | {}'.format(epoch, ' '.join(x for x in m_rate), ' '.join(x for x in m_count)))

    with open(model_file + '.label{}.test'.format(align_weight), 'w')as f:
        [f.write('{}\n'.format(l)) for l in labels]
    if multi:
        with open(model_file + '.hypo{}.test'.format(align_weight), 'w')as f:
            [f.write(o + '\n') for o in outputs]
        with open(model_file + '.align{}.test'.format(align_weight), 'w')as f:
            [f.write('{}\n'.format(a)) for a in alignments]


if __name__ == '__main__':
    main()