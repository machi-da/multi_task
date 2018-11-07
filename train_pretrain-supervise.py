import argparse
import configparser
import os
import shutil
import logging
from logging import getLogger
import numpy as np
import traceback

import convert
import dataset
import evaluate
import gridsearch
import model_supervise
import word2vec

np.set_printoptions(precision=3)
os.environ['CHAINER_TYPE_CHECK'] = '0'
import chainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--pretrain_epoch', '-pe', type=int, default=10)
    parser.add_argument('--interval', '-i', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--vocab', '-v', choices=['normal', 'subword'], default='normal')
    parser.add_argument('--pretrain_w2v', '-p', action='store_true')
    parser.add_argument('--data_path', '-d', choices=['local', 'server'], default='server')
    parser.add_argument('--load_model', '-l', type=str, default='')
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
    vocab_type = args.vocab
    pretrain_w2v = args.pretrain_w2v
    data_path = args.data_path
    load_model = args.load_model

    """DIR PREPARE"""
    config.read(config_file)
    vocab_size = int(config['Parameter']['vocab_size'])

    vocab_name = vocab_type
    if pretrain_w2v:
        vocab_name = 'p' + vocab_name

    model_dir = './pretrain-super_{}{}_{}/'.format(vocab_name, vocab_size, data_path[0])

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
    train_src_file = config[data_path]['train_src_file']
    train_trg_file = config[data_path]['train_trg_file']
    valid_src_file = config[data_path]['valid_src_file']
    valid_trg_file = config[data_path]['valid_trg_file']
    test_src_file = config[data_path]['single_src_file']
    test_trg_file = config[data_path]['single_trg_file']
    # raw_score_file = config[data_path]['raw_score_single_file']
    # raw_score_data = dataset.load_score_file(raw_score_file)
    src_w2v_file = config[data_path]['src_w2v_file']
    trg_w2v_file = config[data_path]['trg_w2v_file']

    train_data_size = dataset.data_size(train_src_file)
    valid_data_size = dataset.data_size(valid_src_file)
    logger.info('pre-train size: {}, pre-valid size: {}'.format(train_data_size, valid_data_size))

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
        src_vocab_size = len(src_vocab.vocab)
        trg_vocab_size = len(trg_vocab.vocab)

        if pretrain_w2v:
            w2v = word2vec.Word2Vec()
            src_initialW, vector_size, src_match_word_count = w2v.make_initialW(src_vocab.vocab, src_w2v_file)
            trg_initialW, vector_size, trg_match_word_count = w2v.make_initialW(trg_vocab.vocab, trg_w2v_file)
            embed_size = vector_size
            hidden_size = vector_size
            logger.info('Initialize w2v embedding. Match: src {}/{}, trg {}/{}'.format(src_match_word_count, src_vocab_size, trg_match_word_count, trg_vocab_size))

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

    train_iter = dataset.Iterator(train_src_file, train_trg_file, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True)
    # train_iter = dataset.Iterator(train_src_file, train_trg_file, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
    valid_iter = dataset.Iterator(valid_src_file, valid_trg_file, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

    """MODEL"""
    model = model_supervise.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW, trg_initialW)

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
    if load_model == '':
        logger.info('Pre-train start')
        pretrain_loss_dic = {}
        for epoch in range(1, pretrain_epoch + 1):
            train_loss = 0
            for i, batch in enumerate(train_iter.generate(), start=1):
                try:
                    loss = model.pretrain(*batch)
                    train_loss += loss.data
                    optimizer.target.cleargrads()
                    loss.backward()
                    optimizer.update()

                except Exception as e:
                    logger.info('P{} ## iteration: {}, {}'.format(epoch, i, e))
                    with open(model_dir + 'error_log.txt', 'a')as f:
                        f.write('P{} ## iteration {}\n'.format(epoch, i))
                        f.write(traceback.format_exc())
                        for b in batch[0]:
                            [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]
            chainer.serializers.save_npz(model_dir + 'p_model_epoch_{}.npz'.format(epoch), model)

            """EVALUATE"""
            valid_loss = 0
            for batch in valid_iter.generate():
                with chainer.no_backprop_mode(), chainer.using_config('train', False):
                    valid_loss += model.pretrain(*batch).data
            logger.info('P{} ## train loss: {}, val loss:{}'.format(epoch, train_loss, valid_loss))
            pretrain_loss_dic[epoch] = valid_loss

        """MODEL SAVE & LOAD"""
        best_epoch = min(pretrain_loss_dic, key=(lambda x: pretrain_loss_dic[x]))
        logger.info('best_epoch:{}'.format(best_epoch))
        chainer.serializers.save_npz(model_dir + 'p_best_model.npz', model)
        logger.info('Pre-train finish')

    """MAIN"""
    valid_num = 5
    label_data, src_data = dataset.load_with_label_binary(test_src_file)
    trg_data = dataset.load(test_trg_file)
    correct_label_data, _, correct_index_data = dataset.load_with_label_index(test_src_file)
    slice_size = len(label_data) // valid_num

    label_data, src_data, trg_data, correct_label_data, correct_index_data = \
        gridsearch.shuffle_list(label_data, src_data, trg_data, correct_label_data, correct_index_data)

    split_label = gridsearch.slice_list(label_data, slice_size)
    split_src = gridsearch.slice_list(src_data, slice_size)
    split_trg = gridsearch.slice_list(trg_data, slice_size)
    split_correct_label = gridsearch.slice_list(correct_label_data, slice_size)
    split_correct_index = gridsearch.slice_list(correct_index_data, slice_size)

    cross_valid_result = []
    s_result_total = []
    for ite in range(1, valid_num + 1):

        model_valid_dir = model_dir + 'valid{}/'.format(ite)
        if not os.path.exists(model_valid_dir):
            os.mkdir(model_valid_dir)

        l_train, l_dev, l_test = gridsearch.split_train_dev_test(split_label, ite - 1)
        s_train, s_dev, s_test = gridsearch.split_train_dev_test(split_src, ite - 1)
        t_train, t_dev, t_test = gridsearch.split_train_dev_test(split_trg, ite - 1)
        c_train, c_dev, c_test = gridsearch.split_train_dev_test(split_correct_label, ite - 1)
        ci_train, ci_dev, ci_test = gridsearch.split_train_dev_test(split_correct_index, ite - 1)

        logger.info('V{} ## train size: {}, dev size: {},test size: {}'.format(ite, len(t_train), len(t_dev), len(t_test)))

        train_iter = dataset.SuperviseIterator(s_train, l_train, t_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True)
        # train_iter = dataset.SuperviseIterator(s_train, l_train, t_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        dev_iter = dataset.SuperviseIterator(s_dev, l_dev, t_dev, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        test_iter = dataset.SuperviseIterator(s_test, l_test, t_test, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

        evaluater = evaluate.Evaluate()

        """MODEL"""
        model = model_supervise.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW, trg_initialW)
        chainer.serializers.load_npz(model_dir + 'p_best_model.npz', model)

        """OPTIMIZER"""
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        """GPU"""
        if gpu_id >= 0:
            model.to_gpu()

        """TRAIN"""
        accuracy_dic = {}
        s_result_dic = {}
        for epoch in range(1, n_epoch + 1):
            train_loss = 0
            for i, batch in enumerate(train_iter.generate(), start=1):
                try:
                    loss = optimizer.target(*batch)
                    train_loss += loss.data
                    optimizer.target.cleargrads()
                    loss.backward()
                    optimizer.update()

                    if i % interval == 0:
                        logger.info('V{} ## E{} ## iteration:{}, loss:{}'.format(ite, epoch, i, train_loss))
                        train_loss = 0

                except Exception as e:
                    logger.info('V{} ## E{} ## train iter: {}, {}'.format(ite, epoch, i, e))
                    with open(model_dir + 'error_log.txt', 'a')as f:
                        f.write('V{} ## E{} ## train iter: {}\n'.format(ite, epoch, i))
                        f.write(traceback.format_exc())
                        f.write('V{} ## E{} ## [batch detail]'.format(ite, epoch))
                        for b in batch[0]:
                            [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]
            logger.info('V{} ## E{} ## train loss:{}'.format(ite, epoch, train_loss))
            chainer.serializers.save_npz(model_valid_dir + 'model_epoch_{}.npz'.format(epoch), model)

            """DEV"""
            labels = []
            alignments = []
            for i, batch in enumerate(dev_iter.generate(), start=1):
                try:
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        _, label, _ = model.predict(batch[0], sos, eos)
                except Exception as e:
                    logger.info('V{} ## E{} ## dev iter: {}, {}'.format(ite, epoch, i, e))
                    with open(model_dir + 'error_log.txt', 'a')as f:
                        f.write('V{} ## E{} ## dev iter: {}\n'.format(ite, epoch, i))
                        f.write(traceback.format_exc())
                        f.write('V{} ## E{} ## [batch detail]'.format(ite, epoch))
                        for b in batch[0]:
                            [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]

                for l in label:
                    labels.append(chainer.cuda.to_cpu(l))

            best_param_dic = evaluater.param_search(labels, alignments, c_dev)

            k = max(best_param_dic, key=lambda x: best_param_dic[x])
            v = best_param_dic[k]
            logger.info('V{} ## E{} ## dev tuning: {}, {}'.format(ite, epoch, k, v))

            """TEST"""
            labels = []
            alignments = []
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

            init, mix = evaluate.key_to_param(k)
            s_rate, s_count, m_rate, m_count, s_result = evaluater.eval_param(labels, alignments, c_test, ci_test, init, mix)
            logger.info('V{} ## E{} ## {}'.format(ite, epoch, ' '.join(s_rate)))
            # logger.info('V{} ## E{} ## {}'.format(ite, epoch, ' '.join(s_count)))

            with open(model_valid_dir + 'model_epoch_{}.label'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(l)) for l in labels]

            accuracy_dic[epoch] = [float(v), s_rate]

        """MODEL SAVE"""
        best_epoch = max(accuracy_dic, key=(lambda x: accuracy_dic[x][0]))
        s_result_total.extend(s_result_dic[best_epoch])
        cross_valid_result.append([ite, best_epoch, accuracy_dic[best_epoch][1]])
        logger.info('V{} ## best_epoch:{} {}'.format(ite, best_epoch, model_valid_dir))
        chainer.serializers.save_npz(model_valid_dir + 'best_model.npz', model)
        shutil.copy(model_valid_dir + 'model_epoch_{}.label'.format(epoch), model_valid_dir + 'best_model.label')

        logger.info('')

    average_score = 0
    for r in cross_valid_result:
        average_score += float(r[2][-1])
        logger.info('{}: epoch{}, {}'.format(r[0], r[1], ' '.join(r[2])))
    average_score /= len(cross_valid_result)
    logger.info('average score: {}'.format(average_score))

    with open(model_dir + 's_res.txt', 'w')as f:
        [f.write('{}\t{}\n'.format(l[0], l[1])) for l in sorted(s_result_total, key=lambda x: x[0])]


if __name__ == '__main__':
    main()