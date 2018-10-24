import argparse
import configparser
import os
import re
import shutil
import logging
from logging import getLogger
import numpy as np
import traceback

import convert
import dataset
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
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--interval', '-i', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--vocab', '-v', choices=['normal', 'subword'], default='normal')
    parser.add_argument('--pretrain_w2v', '-p', action='store_true')
    parser.add_argument('--data_path', '-d', choices=['local', 'server'], default='server')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    """ARGS DETAIL"""
    config_file = args.config_file
    batch_size = args.batch
    n_epoch = args.epoch
    interval = args.interval
    gpu_id = args.gpu
    vocab_type = args.vocab
    pretrain_w2v = args.pretrain_w2v
    data_path = args.data_path

    """DIR PREPARE"""
    config.read(config_file)
    vocab_size = int(config['Parameter']['vocab_size'])
    base_dir = config[data_path]['base_dir']
    dir_path_last = re.search(r'.*/(.*?)$', base_dir).group(1)

    vocab_name = vocab_type
    if pretrain_w2v:
        vocab_name = 'p' + vocab_name

    model_dir = './supervise_{}{}_{}_{}/'.format(vocab_name, vocab_size, data_path[0], dir_path_last)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    shutil.copyfile(config_file, model_dir + config_file)
    config_file = model_dir + config_file
    config.read(config_file)

    """PARAMATER"""
    embed_size = int(config['Parameter']['embed_size'])
    hidden_size = int(config['Parameter']['hidden_size'])
    class_size = 2
    dropout_ratio = float(config['Parameter']['dropout'])
    weight_decay = float(config['Parameter']['weight_decay'])
    gradclip = float(config['Parameter']['gradclip'])
    vocab_size = int(config['Parameter']['vocab_size'])
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

    logger.info(args)  # 引数を記録
    logger.info('[Training start] logging to {}'.format(log_file))

    """DATASET"""
    test_src_file = config[data_path]['test_src_file']

    src_initialW = None

    valid_num = 2
    train_label, train_text = dataset.load_with_label_binary(test_src_file)
    correct_label, _ = dataset.load_with_label_index(test_src_file)
    slice_size = len(train_label) // valid_num

    train_label, train_text, correct_label = gridsearch.shuffle_list(train_label, train_text, correct_label)

    train_label = gridsearch.slice_list(train_label, slice_size)
    train_text = gridsearch.slice_list(train_text, slice_size)
    correct_label = gridsearch.slice_list(correct_label, slice_size)

    for ite in range(1, len(train_label) + 1):

        model_valid_dir = model_dir + 'valid{}/'.format(ite)
        if not os.path.exists(model_valid_dir):
            os.mkdir(model_valid_dir)

        l_dev, l_test = gridsearch.split_dev_test(train_label, ite - 1)
        t_dev, t_test = gridsearch.split_dev_test(train_text, ite - 1)
        c_dev, c_test = gridsearch.split_dev_test(correct_label, ite - 1)

        if vocab_type == 'normal':
            src_vocab = dataset.VocabNormal()
            if os.path.isfile(model_dir + 'src_vocab.normal.pkl'):
                src_vocab.load(model_dir + 'src_vocab.normal.pkl')
            else:
                init_vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
                src_vocab.build(t_dev, False, init_vocab, vocab_size)
                dataset.save_pickle(model_valid_dir + 'src_vocab.normal.pkl', src_vocab.vocab)
            src_vocab.set_reverse_vocab()

            sos = convert.convert_list(np.array([src_vocab.vocab['<s>']], dtype=np.int32), gpu_id)
            eos = convert.convert_list(np.array([src_vocab.vocab['</s>']], dtype=np.int32), gpu_id)
            src_vocab_size = len(src_vocab.vocab)

        #     if pretrain_w2v:
        #         w2v = word2vec.Word2Vec()
        #         src_initialW, vector_size, src_match_word_count = w2v.make_initialW(src_vocab.vocab, src_w2v_file)
        #         embed_size = vector_size
        #         hidden_size = vector_size
        #         logger.info('Initialize w2v embedding. Match: src {}/{}'.format(src_match_word_count, src_vocab_size))
        #
        # elif vocab_type == 'subword':
        #     src_vocab = dataset.VocabSubword()
        #     trg_vocab = dataset.VocabSubword()
        #     if os.path.isfile(model_dir + 'src_vocab.sub.model'):
        #         src_vocab.load(model_dir + 'src_vocab.sub.model')
        #     else:
        #         src_vocab.build(train_src_file + '.sub', model_dir + 'src_vocab.sub', vocab_size)
        #
        #     sos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('<s>')], dtype=np.int32), gpu_id)
        #     eos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('</s>')], dtype=np.int32), gpu_id)
        #     src_vocab_size = len(src_vocab.vocab)

        logger.info('V{} ## train size: {}, test size: {}, src_vocab size: {}'.format(ite, len(t_dev), len(t_test), src_vocab_size))

        train_iter = dataset.SuperviseIterator(t_dev, l_dev, src_vocab, batch_size, gpu_id, sort=True, shuffle=True)
        # train_iter = dataset.SuperviseIterator(t_dev, l_dev, src_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        test_iter = dataset.SuperviseIterator(t_test, l_test, src_vocab, batch_size, gpu_id, sort=False, shuffle=False)

        gridsearcher = gridsearch.GridSearch(test_src_file, valid_num=2)

        """MODEL"""
        model = model_reg.Supervise(src_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW)

        """OPTIMIZER"""
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        """GPU"""
        if gpu_id >= 0:
            # logger.info('Use GPU')
            chainer.cuda.get_device_from_id(gpu_id).use()
            model.to_gpu()

        """TRAIN"""
        sum_loss = 0
        accuracy_dic = {}
        for epoch in range(1, n_epoch + 1):
            for i, batch in enumerate(train_iter.generate(), start=1):
                try:
                    loss = optimizer.target(*batch)
                    sum_loss += loss.data
                    optimizer.target.cleargrads()
                    loss.backward()
                    optimizer.update()

                    if i % interval == 0:
                        logger.info('V{} ## E{} ## iteration:{}, loss:{}'.format(ite, epoch, i, sum_loss))
                        sum_loss = 0

                except Exception as e:
                    logger.info('V{} ## E{} ## train iter: {}, {}'.format(ite, epoch, i, e))
                    with open(model_dir + 'error_log.txt', 'a')as f:
                        f.write('V{} ## E{} ## train iter: {}\n'.format(ite, epoch, i))
                        f.write(traceback.format_exc())
                        f.write('V{} ## E{} ## [batch detail]'.format(ite, epoch))
                        for b in batch[0]:
                            [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]
            logger.info('V{} ## E{} ## train loss:{}'.format(ite, epoch, sum_loss))
            sum_loss = 0
            chainer.serializers.save_npz(model_valid_dir + 'model_epoch_{}.npz'.format(epoch), model)

            """TEST"""
            labels = []
            for i, batch in enumerate(test_iter.generate(), start=1):
                try:
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        _, label, _ = model.predict(batch[0], sos, eos)
                except Exception as e:
                    logger.info('V{} ## E{} ## test iter: {}, {}'.format(ite, epoch, i, e))
                    with open(model_dir + 'error_log.txt', 'a')as f:
                        f.write('V{} ## E{} ## test iter: {}\n'.format(ite, epoch, i))
                        f.write(traceback.format_exc())
                        f.write('V{} ## E{} ## [batch detail]'.format(ite, epoch))
                        for b in batch[0]:
                            [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]

                for l in label:
                    labels.append(chainer.cuda.to_cpu(l))
            param, total, s_total, init, mix = gridsearcher.gridsearch(c_test, labels)
            logger.info('V{} ## E{} ## {}'.format(ite, epoch, param))
            logger.info('V{} ## E{} ## {}'.format(ite, epoch, total))
            with open(model_valid_dir + 'model_epoch_{}.label'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(l)) for l in labels]

            accuracy_dic[epoch] = s_total

        """MODEL SAVE"""
        # best_epoch = min(loss_dic, key=(lambda x: loss_dic[x]))
        best_epoch = max(accuracy_dic, key=(lambda x: accuracy_dic[x]))
        logger.info('V{} ## best_epoch:{} {}'.format(ite, best_epoch, model_dir))
        chainer.serializers.save_npz(model_valid_dir + 'best_model.npz', model)

        logger.info('')


if __name__ == '__main__':
    main()