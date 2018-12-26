import argparse
import configparser
import os
import shutil
import numpy as np
import traceback

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
    parser.add_argument('--batch', '-b', type=int, default=30)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--pretrain_epoch', '-pe', type=int, default=10)
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
    gpu_id = args.gpu
    model_type = args.model
    vocab_type = args.vocab
    pretrain_w2v = args.pretrain_w2v
    data_path = args.data_path

    """DIR PREPARE"""
    config.read(config_file)
    vocab_size = int(config['Parameter']['vocab_size'])
    coefficient = float(config['Parameter']['coefficient'])

    vocab_name = vocab_type
    if pretrain_w2v:
        vocab_name = 'p' + vocab_name

    if model_type == 'multi':
        model_dir = './super_{}_{}{}_{}_c{}/'.format(model_type, vocab_name, vocab_size, data_path[0], coefficient)
    else:
        model_dir = './super_{}_{}{}_{}/'.format(model_type, vocab_name, vocab_size, data_path[0])

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
    test_src_file = config[data_path]['single_src_file']
    test_trg_file = config[data_path]['single_trg_file']
    src_w2v_file = config[data_path]['src_w2v_file']
    trg_w2v_file = config[data_path]['trg_w2v_file']

    correct_label, src_label, src_text, correct_index = dataset.load_binary_score_file(test_src_file)
    trg_text = dataset.load(test_trg_file)
    slice_size = len(correct_label) // valid_num
    correct_label, src_label, src_text, trg_text, correct_index = gridsearch.shuffle_list(correct_label, src_label, src_text, trg_text, correct_index)

    correct_label = gridsearch.slice_list(correct_label, slice_size)
    src_label = gridsearch.slice_list(src_label, slice_size)
    src_text = gridsearch.slice_list(src_text, slice_size)
    trg_text = gridsearch.slice_list(trg_text, slice_size)
    correct_index = gridsearch.slice_list(correct_index, slice_size)

    evaluater = evaluate.Evaluate()

    cross_valid_result = []
    s_result_total = []
    for ite in range(1, valid_num + 1):
        model_valid_dir = model_dir + 'valid{}/'.format(ite)
        if not os.path.exists(model_valid_dir):
            os.mkdir(model_valid_dir)

        index = ite - 1
        c_label_train, c_label_dev, c_label_test = gridsearch.split_train_dev_test(correct_label, index)
        label_train, label_dev, label_test = gridsearch.split_train_dev_test(src_label, index)
        src_train, src_dev, src_test = gridsearch.split_train_dev_test(src_text, index)
        trg_train, trg_dev, trg_test = gridsearch.split_train_dev_test(trg_text, index)
        c_index_train, c_index_dev, c_index_test = gridsearch.split_train_dev_test(correct_index, index)

        """VOCABULARY"""
        src_vocab, trg_vocab, sos, eos = dataset.prepare_vocab(model_valid_dir, vocab_type, src_train, trg_train, vocab_size, gpu_id)
        src_vocab_size = len(src_vocab.vocab)
        trg_vocab_size = len(trg_vocab.vocab)

        src_initialW = None
        trg_initialW = None

        if pretrain_w2v:
            w2v = word2vec.Word2Vec()
            src_initialW, vector_size, src_match_word_count = w2v.make_initialW(src_vocab.vocab, src_w2v_file)
            trg_initialW, vector_size, trg_match_word_count = w2v.make_initialW(trg_vocab.vocab, trg_w2v_file)
            logger.info('Initialize w2v embedding. Match: src {}/{}, trg {}/{}'.format(src_match_word_count, src_vocab_size, trg_match_word_count, trg_vocab_size))

        """ITERATOR"""
        train_iter = dataset.Iterator(src_train, label_train, trg_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True)
        # train_iter = dataset.Iterator(src_train, label_train, trg_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        dev_iter = dataset.Iterator(src_dev, label_dev, trg_dev, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        test_iter = dataset.Iterator(src_test, label_test, trg_test, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

        logger.info('V{} ## train: {}, dev: {},test: {}, src_vocab: {}, trg_vocab: {}'.format(ite, len(label_train), len(label_dev), len(label_test), src_vocab_size, trg_vocab_size))

        """MODEL"""
        if model_type == 'multi':
            model = model_supervise.Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, src_initialW, trg_initialW)
        elif model_type in ['label', 'pretrain']:
            model = model_supervise.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW, trg_initialW)
        else:
            model = model_supervise.EncoderDecoder(src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_ratio, src_initialW, trg_initialW)

        """OPTIMIZER"""
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        """GPU"""
        if gpu_id >= 0:
            chainer.cuda.get_device_from_id(gpu_id).use()
            model.to_gpu()

        """PRETRAIN"""
        if model_type == 'pretrain':
            logger.info('Pre-train start')
            logger.info('train size: {}, valid size: {}'.format(len(label_train), len(label_dev)))
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
                        logger.info('V{} ## P{} ## train iter: {}, {}'.format(ite, epoch, i, e))

                chainer.serializers.save_npz(model_valid_dir + 'p_model_epoch_{}.npz'.format(epoch), model)

                """EVALUATE"""
                valid_loss = 0
                for batch in dev_iter.generate():
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        valid_loss += model.pretrain(*batch).data
                logger.info('V{} ## P{} ## train loss: {}, val loss:{}'.format(ite, epoch, train_loss, valid_loss))
                pretrain_loss_dic[epoch] = valid_loss

            """MODEL SAVE"""
            best_epoch = min(pretrain_loss_dic, key=(lambda x: pretrain_loss_dic[x]))
            logger.info('best_epoch:{}, val loss: {}'.format(best_epoch, pretrain_loss_dic[best_epoch]))
            shutil.copyfile(model_valid_dir + 'p_model_epoch_{}.npz'.format(best_epoch), model_valid_dir + 'p_best_model.npz')
            logger.info('Pre-train finish')

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

                except Exception as e:
                    logger.info('V{} ## E{} ## train iter: {}, {}'.format(ite, epoch, i, e))
            chainer.serializers.save_npz(model_valid_dir + 'model_epoch_{}.npz'.format(epoch), model)

            """DEV"""
            outputs = []
            labels = []
            alignments = []
            for i, batch in enumerate(dev_iter.generate(), start=1):
                try:
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        output, label, align = model.predict(batch[0], sos, eos)
                except Exception as e:
                    logger.info('V{} ## E{} ## dev iter: {}, {}'.format(ite, epoch, i, e))

                if model_type == 'multi':
                    for o, l, a in zip(output, label, align):
                        outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
                        labels.append(chainer.cuda.to_cpu(l))
                        alignments.append(chainer.cuda.to_cpu(a))
                elif model_type in ['label', 'pretrain']:
                    for l in label:
                        labels.append(chainer.cuda.to_cpu(l))
                else:
                    for o, a in zip(output, align):
                        outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
                        alignments.append(chainer.cuda.to_cpu(a))

            best_param_dic = evaluater.param_search(labels, alignments, c_label_dev)
            param = max(best_param_dic, key=lambda x: best_param_dic[x])
            init, mix = evaluate.key_to_param(param)
            dev_score = round(best_param_dic[param], 3)

            """TEST"""
            outputs = []
            labels = []
            alignments = []
            for i, batch in enumerate(test_iter.generate(), start=1):
                try:
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        output, label, align = model.predict(batch[0], sos, eos)
                except Exception as e:
                    logger.info('V{} ## E{} ## test iter: {}, {}'.format(ite, epoch, i, e))
                if model_type == 'multi':
                    for o, l, a in zip(output, label, align):
                        outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
                        labels.append(chainer.cuda.to_cpu(l))
                        alignments.append(chainer.cuda.to_cpu(a))
                elif model_type in ['label', 'pretrain']:
                    for l in label:
                        labels.append(chainer.cuda.to_cpu(l))
                else:
                    for o, a in zip(output, align):
                        outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
                        alignments.append(chainer.cuda.to_cpu(a))

            if model_type in ['multi', 'label', 'pretrain']:
                s_rate, s_count, _, _, s_result = evaluater.eval_param(labels, alignments, c_label_test, c_index_test, init, mix)
            else:
                s_rate, s_count, _, _, s_result = evaluater.eval_param(alignments, [], c_label_test, c_index_test, init, mix)
            test_score = round(s_rate[-1], 3)
            s_result_dic[epoch] = s_result
            logger.info('V{} ## E{} ## loss:{}, dev: {}, test: {}'.format(ite, epoch, train_loss, dev_score, test_score))

            dataset.save_output(model_valid_dir, epoch, labels, alignments, outputs, s_result)
            accuracy_dic[epoch] = [dev_score, test_score]

        """MODEL SAVE"""
        best_epoch = max(accuracy_dic, key=(lambda x: accuracy_dic[x][0]))
        s_result_total.extend(s_result_dic[best_epoch])
        cross_valid_result.append([ite, best_epoch, accuracy_dic[best_epoch][1]])
        logger.info('V{} ## best_epoch:{}, dev:{}, test:{}'.format(ite, best_epoch, accuracy_dic[best_epoch][0], accuracy_dic[best_epoch][1]))
        shutil.copyfile(model_valid_dir + 'model_epoch_{}.npz'.format(best_epoch), model_valid_dir + 'best_model.npz')

        logger.info('')

    average_score = [0 for _ in range(len(cross_valid_result[0]))]
    for r in cross_valid_result:
        average_score = [average_score[i] + r[2][i] for i in range(len(average_score))]
        logger.info('\t{}: epoch{}, {}'.format(r[0], r[1], ' '.join(dataset.float_to_str(r[2]))))
    average_score = [average_score[i] / len(cross_valid_result) for i in range(len(average_score))]
    logger.info('ave: {}'.format(' '.join(dataset.float_to_str(average_score))))

    with open(model_dir + 's_res.txt', 'w')as f:
        [f.write('{}\n'.format(l[1])) for l in sorted(s_result_total, key=lambda x: x[0])]


if __name__ == '__main__':
    main()