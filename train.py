import argparse
import configparser
import os
import re
import shutil
import numpy as np
import traceback

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
    parser.add_argument('--pretrain_epoch', '-pe', type=int, default=5)
    parser.add_argument('--interval', '-i', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', choices=['multi', 'label', 'encdec', 'pretrain'], default='multi')
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
        model_dir = './{}_{}{}_{}_c{}_{}/'.format(model_type, vocab_name, vocab_size, data_path[0], coefficient, dir_path_last)
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
    valid_num = int(config['Parameter']['valid_num'])
    """LOGGER"""
    log_file = model_dir + 'log.txt'
    logger = dataset.prepare_logger(log_file)

    logger.info(args)  # 引数を記録
    logger.info('[Training start] logging to {}'.format(log_file))

    """DATASET"""
    train_src_file = config[data_path]['train_src_file']
    train_trg_file = config[data_path]['train_trg_file']
    valid_src_file = config[data_path]['valid_src_file']
    valid_trg_file = config[data_path]['valid_trg_file']
    test_src_file = config[data_path]['test_src_file']
    raw_score_file = config[data_path]['raw_score_file']
    raw_score = dataset.load_score_file(raw_score_file)
    src_w2v_file = config[data_path]['src_w2v_file']
    trg_w2v_file = config[data_path]['trg_w2v_file']

    correct_label, _, correct_index = dataset.load_with_label_index(test_src_file)

    train_data_size = dataset.data_size(train_src_file)
    valid_data_size = dataset.data_size(valid_src_file)
    logger.info('train size: {}, valid size: {}'.format(train_data_size, valid_data_size))

    """VOCABULARY"""
    src_vocab, trg_vocab, sos, eos = dataset.prepare_vocab(model_dir, vocab_type, train_src_file, train_trg_file, vocab_size, gpu_id)
    src_vocab_size = len(src_vocab.vocab)
    trg_vocab_size = len(trg_vocab.vocab)

    src_initialW = None
    trg_initialW = None

    if pretrain_w2v:
        w2v = word2vec.Word2Vec()
        src_initialW, vector_size, src_match_word_count = w2v.make_initialW(src_vocab.vocab, src_w2v_file)
        trg_initialW, vector_size, trg_match_word_count = w2v.make_initialW(trg_vocab.vocab, trg_w2v_file)
        logger.info('Initialize w2v embedding. Match: src {}/{}, trg {}/{}'.format(src_match_word_count, src_vocab_size, trg_match_word_count, trg_vocab_size))

    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    """ITERATOR"""
    src_label, src_text = dataset.load_with_label_reg(train_src_file)
    trg_text = dataset.load(train_trg_file)
    train_iter = dataset.Iterator(src_text, src_label, trg_text, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True)
    # train_iter = dataset.Iterator(train_src_file, train_trg_file, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

    src_label, src_text = dataset.load_with_label_reg(valid_src_file)
    trg_text = dataset.load(valid_trg_file)
    valid_iter = dataset.Iterator(src_text, src_label, trg_text, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

    src_label, src_text = dataset.load_with_label_reg(test_src_file)
    trg_text = dataset.load(test_src_file)
    test_iter = dataset.Iterator(src_text, src_label, trg_text, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

    gridsearcher = gridsearch.GridSearch(valid_num=valid_num)

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
        train_loss = 0
        pretrain_loss_dic = {}
        for epoch in range(1, pretrain_epoch + 1):
            for i, batch in enumerate(train_iter.generate(), start=1):
                try:
                    loss = model.pretrain(*batch)
                    train_loss += loss.data
                    optimizer.target.cleargrads()
                    loss.backward()
                    optimizer.update()

                    if i % interval == 0:
                        logger.info('P{} ## iteration:{}, loss:{}'.format(epoch, i, train_loss))
                        train_loss = 0

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
            logger.info('P{} ## val loss:{}'.format(epoch, valid_loss))
            pretrain_loss_dic[epoch] = valid_loss

        """MODEL SAVE & LOAD"""
        best_epoch = min(pretrain_loss_dic, key=(lambda x: pretrain_loss_dic[x]))
        logger.info('best_epoch:{}'.format(best_epoch))
        chainer.serializers.save_npz(model_dir + 'p_best_model.npz', model)
        logger.info('Pre-train finish')

    """TRAIN"""
    accuracy_dic = {}
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
                    logger.info('E{} ## iteration:{}, loss:{}'.format(epoch, i, train_loss))
                    train_loss = 0

            except Exception as e:
                logger.info('E{} ## train iter: {}, {}'.format(epoch, i, e))
                with open(model_dir + 'error_log.txt', 'a')as f:
                    f.write('E{} ## train iter: {}\n'.format(epoch, i))
                    f.write(traceback.format_exc())
                    f.write('E{} ## [batch detail]'.format(epoch))
                    for b in batch[0]:
                        [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]
        chainer.serializers.save_npz(model_dir + 'model_epoch_{}.npz'.format(epoch), model)

        """EVALUATE"""
        valid_loss = 0
        for i, batch in enumerate(valid_iter.generate(), start=1):
            try:
                with chainer.no_backprop_mode(), chainer.using_config('train', False):
                    valid_loss += optimizer.target(*batch).data
            except Exception as e:
                logger.info('E{} ## valid iter: {}, {}'.format(epoch, i, e))
                with open(model_dir + 'error_log.txt', 'a')as f:
                    f.write('E{} ## valid iter: {}\n'.format(epoch, i))
                    f.write(traceback.format_exc())
                    f.write('E{} ## [batch detail]'.format(epoch))
                    for b in batch[0]:
                        [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]

        logger.info('E{} ## train loss: {}, val loss:{}'.format(epoch, train_loss, valid_loss))

        """TEST"""
        outputs = []
        labels = []
        alignments = []
        for i, batch in enumerate(test_iter.generate(), start=1):
            try:
                with chainer.no_backprop_mode(), chainer.using_config('train', False):
                    output, label, align = model.predict(batch[0], sos, eos)
            except Exception as e:
                logger.info('E{} ## test iter: {}, {}'.format(epoch, i, e))
                with open(model_dir + 'error_log.txt', 'a')as f:
                    f.write('E{} ## test iter: {}\n'.format(epoch, i))
                    f.write(traceback.format_exc())
                    f.write('E{} ## [batch detail]'.format(epoch))
                    for b in batch[0]:
                        [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]

            for o in output:
                outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
            for l in label:
                labels.append(chainer.cuda.to_cpu(l))
            for a in align:
                alignments.append(chainer.cuda.to_cpu(a))

        if model_type in ['multi', 'label', 'pretrain']:
            _, total, s_total, s_result_total, dev_score = gridsearcher.gridsearch(correct_label, correct_index, labels, alignments)
        else:
            _, total, s_total, s_result_total, dev_score = gridsearcher.gridsearch(correct_label, correct_index, raw_score, alignments)
        logger.info('E{} ## dev: {}, {}'.format(epoch, round(dev_score, 3), total))

        dataset.save_output(model_dir, epoch, labels, alignments, outputs)
        accuracy_dic[epoch] = s_total

        with open(model_dir + 'model_epoch_{}.s_res.txt'.format(epoch), 'w')as f:
            [f.write('{},{}\n'.format(l[0], l[1])) for l in sorted(s_result_total, key=lambda x: x[0])]

    """MODEL SAVE"""
    best_epoch = max(accuracy_dic, key=(lambda x: accuracy_dic[x]))
    logger.info('best_epoch:{}, score: {}, {}'.format(best_epoch, accuracy_dic[best_epoch], model_dir))
    chainer.serializers.save_npz(model_dir + 'best_model.npz', model)


if __name__ == '__main__':
    main()