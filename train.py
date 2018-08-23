import argparse
import configparser
import os
import re
import logging
import shutil
from logging import getLogger
import numpy as np
import traceback

import convert
import dataset
import evaluate
import gridsearch
import model_reg
import word2vec

np.set_printoptions(precision=3)
os.environ['CHAINER_TYPE_CHECK'] = '0'
import chainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--pretrain_epoch', '-pe', type=int, default=5)
    parser.add_argument('--interval', '-i', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', choices=['multi', 'label', 'encdec', 'pretrain'], default='multi')
    parser.add_argument('--vocab', '-v', choices=['normal', 'subword'], default='normal')
    parser.add_argument('--pretrain_w2v', '-p', action='store_true')
    parser.add_argument('--data_path', '-d', choices=['local', 'server'], default='local')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    """ARGS DETAIL"""
    config_file = args.config_file
    batch_size = args.batch
    n_epoch = args.epoch
    pretrain_epoch = args.pretrain_epoch
    interval = args.interval
    gpu_id = args.gpu
    model_type = args.model
    vocab_type = args.vocab
    pretrain_w2v = args.pretrain_w2v
    data_path = args.data_path

    """DIR PREPARE"""
    config.read(config_file)
    vocab_size = int(config['Parameter']['vocab_size'])
    coefficient = float(config['Parameter']['coefficient'])
    base_dir = config[data_path]['base_dir']
    dir_path_last = re.search(r'.*/(.*?)$', base_dir).group(1)

    vocab_name = vocab_type
    if pretrain_w2v:
        vocab_name = 'p' + vocab_name

    if model_type == 'multi':
        model_dir = './{}_{}{}_{}_c{}_{}/'.format(model_type, vocab_name, vocab_size,data_path[0], coefficient, dir_path_last)
    else:
        model_dir = './{}_{}{}_{}_{}/'.format(model_type, vocab_name, vocab_size, data_path[0], dir_path_last)

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
    train_src_file = config[data_path]['train_src_file']
    train_trg_file = config[data_path]['train_trg_file']
    valid_src_file = config[data_path]['valid_src_file']
    valid_trg_file = config[data_path]['valid_trg_file']
    test_src_file = config[data_path]['test_src_file']
    row_score_file = config[data_path]['row_score_file']
    row_score = dataset.txt_to_list(row_score_file)

    train_data_size = dataset.data_size(train_src_file)
    valid_data_size = dataset.data_size(valid_src_file)
    logger.info('train size: {}, valid size: {}'.format(train_data_size, valid_data_size))

    src_initialW = None
    trg_initialW = None

    if vocab_type == 'normal':
        src_vocab = dataset.VocabNormal()
        trg_vocab = dataset.VocabNormal()
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

        if pretrain_w2v:
            embed_size = 200
            hidden_size = 200
            src_initialW = word2vec.make_initialW(src_vocab.vocab, embed_size)
            trg_initialW = word2vec.make_initialW(trg_vocab.vocab, embed_size)

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

    train_iter = dataset.Iterator(train_src_file, train_trg_file, src_vocab, trg_vocab, batch_size, sort=True, shuffle=True)
    # train_iter = dataset.Iterator(train_src_file, train_trg_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False)
    valid_iter = dataset.Iterator(valid_src_file, valid_trg_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False)
    test_iter = dataset.Iterator(test_src_file, test_src_file, src_vocab, trg_vocab, batch_size, sort=False, shuffle=False)

    evaluater = evaluate.Evaluate(test_src_file)
    gridsearcher = gridsearch.GridSearch(test_src_file)
    """MODEL"""
    if model_type == 'multi':
        model = model_reg.Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, src_initialW, trg_initialW)
    elif model_type in ['label', 'pretrain']:
        model = model_reg.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW, trg_initialW)
    else:
        model = model_reg.EncoderDecoder(src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_ratio, src_initialW, trg_initialW)
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

    """PRETRAIN"""
    if model_type == 'pretrain':
        logger.info('Pre-train start')
        sum_loss = 0
        pretrain_loss_dic = {}
        for epoch in range(1, pretrain_epoch + 1):
            for i, batch in enumerate(train_iter.generate(), start=1):
                try:
                    batch = convert.convert(batch, gpu_id)
                    loss = model.pretrain(*batch)
                    sum_loss += loss.data
                    optimizer.target.cleargrads()
                    loss.backward()
                    optimizer.update()

                    if i % interval == 0:
                        logger.info('P{} ## iteration:{}, loss:{}'.format(epoch, i, sum_loss))
                        sum_loss = 0

                except Exception as e:
                    logger.info(traceback.format_exc())
                    logger.info('iteration: {}'.format(i))
                    with open(model_dir + 'error_log.txt', 'a')as f:
                        f.write('iteration_{}\n'.format(i))
                        for b in batch[0]:
                            for bb in b:
                                f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n')
            chainer.serializers.save_npz(model_dir + 'p_model_epoch_{}.npz'.format(epoch), model)

            """EVALUATE"""
            valid_loss = 0
            for batch in valid_iter.generate():
                batch = convert.convert(batch, gpu_id)
                with chainer.no_backprop_mode(), chainer.using_config('train', False):
                    valid_loss += model.pretrain(*batch).data
            logger.info('P{} ## val loss:{}'.format(epoch, valid_loss))
            pretrain_loss_dic[epoch] = valid_loss

        """MODEL SAVE & LOAD"""
        best_epoch = min(pretrain_loss_dic, key=(lambda x: pretrain_loss_dic[x]))
        logger.info('best_epoch:{}'.format(best_epoch))
        chainer.serializers.save_npz(model_dir + 'p_best_model.npz', model)
        chainer.serializers.load_npz(model_dir + 'p_best_model.npz', model)
        logger.info('Pre-train finish')

    """TRAIN"""
    sum_loss = 0
    loss_dic = {}
    result = ['epoch,valid_loss']
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
        result.append('{},{}'.format(epoch, valid_loss))

        """TEST"""
        outputs = []
        labels = []
        alignments = []
        for i, batch in enumerate(test_iter.generate(), start=1):
            batch = convert.convert(batch, gpu_id)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                output, label, align = model.predict(batch[0], sos, eos)
            for o in output:
                outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
            for l in label:
                labels.append(chainer.cuda.to_cpu(l))
            for a in align:
                alignments.append(chainer.cuda.to_cpu(a))

        if model_type == 'multi':
            score = gridsearcher.split_data(labels, alignments)
            logger.info('E{} ## {}'.format(epoch, score[0]))
            logger.info('E{} ## {}'.format(epoch, score[1]))
            with open(model_dir + 'model_epoch_{}.label'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(l)) for l in labels]
            with open(model_dir + 'model_epoch_{}.hypo'.format(epoch), 'w')as f:
                [f.write(o + '\n') for o in outputs]
            with open(model_dir + 'model_epoch_{}.align'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(a)) for a in alignments]

        elif model_type in ['label', 'pretrain']:
            s_rate, _, _, _ = evaluater.label(labels)
            s_rate_init, _, _, _ = evaluater.label_init(labels)
            logger.info('E{} ## {}: {}, {}: {}'.format(epoch, 'normal', s_rate[-1], 'init 0.7', s_rate_init[-1]))
            with open(model_dir + 'model_epoch_{}.label'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(l)) for l in labels]

        else:
            score = gridsearcher.split_data(row_score, alignments)
            logger.info('E{} ## {}'.format(epoch, score[0]))
            logger.info('E{} ## {}'.format(epoch, score[1]))
            with open(model_dir + 'model_epoch_{}.hypo'.format(epoch), 'w')as f:
                [f.write(o + '\n') for o in outputs]
            with open(model_dir + 'model_epoch_{}.align'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(a)) for a in alignments]

    """MODEL SAVE"""
    best_epoch = min(loss_dic, key=(lambda x: loss_dic[x]))
    logger.info('best_epoch:{} {}'.format(best_epoch, model_dir))
    chainer.serializers.save_npz(model_dir + 'best_model.npz', model)

    with open(model_dir + 'valid_loss.csv', 'w')as f:
        [f.write(r + '\n') for r in result]


if __name__ == '__main__':
    main()