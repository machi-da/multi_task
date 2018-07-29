import argparse
import configparser
import os
import re
import logging
import shutil
from logging import getLogger
import numpy as np
import traceback

np.set_printoptions(precision=3)

import convert
import dataset
import evaluate
from model import Multi
from model_reg import MultiReg

# os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--interval', '-i', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', choices=['multi', 'nn'], default='multi')
    parser.add_argument('--vocab', '-v', choices=['normal', 'subword'], default='normal')
    parser.add_argument('--type', '-t', choices=['l', 'lr', 's', 'sr'], default='l')
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
    config = configparser.ConfigParser()
    """ARGS DETAIL"""
    gpu_id = args.gpu
    n_epoch = args.epoch
    batch_size = args.batch
    interval = args.interval
    reg = False if args.type == 'l' or args.type == 's' else True
    vocab_type = args.vocab
    section = model_type(args.type)
    config_file = args.config_file
    config.read(config_file)

    """DIR PREPARE"""
    coefficient = float(config['Parameter']['coefficient'])
    multi = True if args.model == 'multi' else False
    base_dir = config[section]['base_dir']
    dir_path_last = re.search(r'.*/(.*?)$', base_dir).group(1)
    result = []
    if multi:
        result.append('epoch,valid_loss,s,m,s_init,m_init,s_align,m_align')
        model_dir = './multi_{}_{}_c{}_{}/'.format(vocab_type, args.type, coefficient, dir_path_last)
    else:
        result.append('epoch,valid_loss,s,m,s_init,m_init')
        model_dir = './nn_{}_{}_{}/'.format(vocab_type, args.type, dir_path_last)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    shutil.copyfile(config_file, model_dir + config_file)
    config_file = model_dir + config_file
    config.read(config_file)

    """PARAMATER"""
    embed_size = int(config['Parameter']['embed_size'])
    hidden_size = int(config['Parameter']['hidden_size'])
    class_size = int(config['Parameter']['class_size'])
    dropout_ratio = float(config['Parameter']['dropout'])
    weight_decay = float(config['Parameter']['weight_decay'])
    gradclip = float(config['Parameter']['gradclip'])
    vocab_size = int(config['Parameter']['vocab_size'])
    coefficient = float(config['Parameter']['coefficient'])
    align_weight = float(config['Parameter']['align_weight'])
    multi = bool(int(config['Parameter']['multi']))
    """LOGGER"""
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file = model_dir + 'log.txt'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('[Training start] logging to {}'.format(log_file))

    """DATASET"""
    train_src_file = config[section]['train_src_file']
    train_trg_file = config[section]['train_trg_file']
    valid_src_file = config[section]['valid_src_file']
    valid_trg_file = config[section]['valid_trg_file']
    test_src_file = config[section]['test_src_file']

    train_data_size = dataset.data_size(train_src_file)
    valid_data_size = dataset.data_size(valid_src_file)
    logger.info('train size: {0}, valid size: {1}'.format(train_data_size, valid_data_size))

    if vocab_type == 'normal':
        src_vocab = dataset.VocabNormal(reg)
        trg_vocab = dataset.VocabNormal(reg)
        if os.path.isfile(model_dir + 'src_vocab.normal.pkl') and os.path.isfile(model_dir + 'trg_vocab.normal.pkl'):
            src_vocab.load(model_dir + 'src_vocab.normal.pkl')
            trg_vocab.load(model_dir + 'trg_vocab.normal.pkl')
        else:
            init_vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
            src_vocab.build(train_src_file, True,  init_vocab, vocab_size)
            trg_vocab.build(train_trg_file, False, init_vocab, vocab_size)
            dataset.save_pickle(model_dir + 'src_vocab.normal.pkl', src_vocab.vocab)
            dataset.save_pickle(model_dir + 'trg_vocab.normal.pkl', trg_vocab.vocab)
        src_vocab.set_reverse_vocab()
        trg_vocab.set_reverse_vocab()

        sos = convert.convert_list(np.array([src_vocab.vocab['<s>']], dtype=np.int32), gpu_id)
        eos = convert.convert_list(np.array([src_vocab.vocab['</s>']], dtype=np.int32), gpu_id)

    elif vocab_type == 'subword':
        src_vocab = dataset.VocabSubword()
        trg_vocab = dataset.VocabSubword()
        if os.path.isfile(model_dir + 'src_vocab.sub.model') and os.path.isfile(model_dir + 'trg_vocab.sub.model'):
            src_vocab.load(model_dir + 'src_vocab.sub.model')
            trg_vocab.load(model_dir + 'trg_vocab.sub.model')
        else:
            src_vocab.build(train_src_file + '.sub', model_dir + 'src_vocab.sub', vocab_size)
            trg_vocab.build(train_trg_file + '.sub', model_dir + 'trg_vocab.sub', vocab_size)

        sos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('<s>')], dtype=np.int32), gpu_id)
        eos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('</s>')], dtype=np.int32), gpu_id)

    src_vocab_size = len(src_vocab.vocab)
    trg_vocab_size = len(trg_vocab.vocab)
    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    train_iter = dataset.Iterator(train_src_file, train_trg_file, src_vocab, trg_vocab, batch_size, sort=True, shuffle=True, reg=reg)
    # train_iter = dataset.Iterator(train_src_file, train_trg_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False, reg=reg)
    valid_iter = dataset.Iterator(valid_src_file, valid_trg_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False, reg=reg)
    evaluater = evaluate.Evaluate(test_src_file, align_weight)
    test_iter = dataset.Iterator(test_src_file, test_src_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False)
    """MODEL"""
    if reg:
        class_size = 1
        model = MultiReg(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, multi=multi)
    else:
        model = Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, multi=multi)
    """OPTIMIZER"""
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
    """TRAIN"""
    sum_loss = 0
    loss_dic = {}
    for epoch in range(1, n_epoch + 1):
        for i, batch in enumerate(train_iter.generate(), start=1):
            try:
                batch = convert.convert(batch, gpu_id)
                loss = optimizer.target(*batch)
                sum_loss += loss.data
                optimizer.target.cleargrads()
                loss.backward()
                optimizer.update()

                if i % interval == 0:
                    logger.info('E{} ## iteration:{}, loss:{}'.format(epoch, i, sum_loss))
                    sum_loss = 0

            except Exception as e:
                logger.info(traceback.format_exc())
                logger.info('iteration: {}'.format(i))
                with open(model_dir + 'error_log.txt', 'a')as f:
                    f.write('iteration_{}\n'.format(i))
                    for b in batch[0]:
                        for bb in b:
                            f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n')
        chainer.serializers.save_npz(model_dir + 'model_epoch_{}.npz'.format(epoch), model)

        """EVALUATE"""
        valid_loss = 0
        for batch in valid_iter.generate():
            batch = convert.convert(batch, gpu_id)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                valid_loss += optimizer.target(*batch).data
        logger.info('E{} ## val loss:{}'.format(epoch, valid_loss))
        loss_dic[epoch] = valid_loss

        """TEST"""
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
        res = '{},{},{},{},{},{}'.format(epoch, valid_loss, s_rate[-1], m_rate[-1], s_rate_i[-1], m_rate_i[-1])
        if multi:
            rank_list = evaluater.rank_init_align(labels, alignments)
            s_rate_a, s_count = evaluater.single(rank_list)
            m_rate_a, m_count = evaluater.multiple(rank_list)
            logger.info('E{} ## normal init align'.format(epoch))
            logger.info('E{} ## s: {} | {}'.format(epoch, ' '.join(x for x in s_rate), ' '.join(x for x in s_count)))
            logger.info('E{} ## m: {} | {}'.format(epoch, ' '.join(x for x in m_rate), ' '.join(x for x in m_count)))
            res = '{},{},{},{},{},{},{},{}'.format(epoch, valid_loss, s_rate[-1], m_rate[-1], s_rate_i[-1], m_rate_i[-1], s_rate_a[-1], m_rate_a[-1])

        with open(model_dir + 'model_epoch_{}.label'.format(epoch), 'w')as f:
            [f.write('{}\n'.format(l)) for l in labels]
        if multi:
            with open(model_dir + 'model_epoch_{}.hypo'.format(epoch), 'w')as f:
                [f.write(o + '\n') for o in outputs]
            with open(model_dir + 'model_epoch_{}.align'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(a)) for a in alignments]

        result.append(res)

    """MODEL SAVE"""
    best_epoch = min(loss_dic, key=(lambda x: loss_dic[x]))
    logger.info('best_epoch:{}'.format(best_epoch))
    chainer.serializers.save_npz(model_dir + 'best_model.npz', model)

    with open(model_dir + 'result.csv', 'w')as f:
        [f.write(r + '\n') for r in result]


if __name__ == '__main__':
    main()