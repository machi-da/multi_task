import argparse
import configparser
import os
import re
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
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--pretrain_epoch', '-pe', type=int, default=10)
    parser.add_argument('--interval', '-i', type=int, default=100000)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', choices=['multi', 'label', 'encdec', 'pretrain'], default='label')
    parser.add_argument('--vocab', '-v', choices=['normal', 'subword'], default='normal')
    parser.add_argument('--pretrain_w2v', '-p', action='store_true')
    parser.add_argument('--data_path', '-d', choices=['local', 'server', 'test'], default='server')
    parser.add_argument('--multiple', type=int, default=1)
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
    multiple= args.multiple

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
        model_dir = './mix_super_{}_{}{}_{}_c{}_{}/'.format(model_type, vocab_name, vocab_size, data_path[0], coefficient, dir_path_last)
    else:
        if multiple == 1:
            model_dir = './mix_super_{}_{}{}_{}_{}/'.format(model_type, vocab_name, vocab_size, data_path[0], dir_path_last)
        else:
            model_dir = './mix_super_{}_{}{}_{}_{}_m{}/'.format(model_type, vocab_name, vocab_size, data_path[0], dir_path_last, multiple)

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

    """MAIN"""
    # QAデータ
    label_data, src_data = dataset.load_with_label_reg(train_src_file)
    trg_data = dataset.load(train_trg_file)
    slice_size = len(label_data) // valid_num
    label_data, src_data, trg_data = gridsearch.shuffle_list(label_data, src_data, trg_data)
    split_qa_label = gridsearch.slice_list(label_data, slice_size)
    split_qa_src = gridsearch.slice_list(src_data, slice_size)
    split_qa_trg = gridsearch.slice_list(trg_data, slice_size)

    # 人手データ
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

    evaluater = evaluate.Evaluate()

    cross_valid_result = []
    s_result_total = []
    for ite in range(1, valid_num + 1):

        model_valid_dir = model_dir + 'valid{}/'.format(ite)
        if not os.path.exists(model_valid_dir):
            os.mkdir(model_valid_dir)

        # QAデータ
        # 18万
        # q_train, _, _ = gridsearch.split_train_dev_test(split_qa_src, ite - 1)
        # ql_train, _, _ = gridsearch.split_train_dev_test(split_qa_label, ite - 1)
        # a_train, _, _ = gridsearch.split_train_dev_test(split_qa_trg, ite - 1)

        # 6万
        _, q_train, _ = gridsearch.split_train_dev_test(split_qa_src, ite - 1)
        _, ql_train, _ = gridsearch.split_train_dev_test(split_qa_label, ite - 1)
        _, a_train, _ = gridsearch.split_train_dev_test(split_qa_trg, ite - 1)

        qa_iter = dataset.Iterator(q_train, ql_train, a_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=True)

        # 人手データ
        l_train, l_dev, l_test = gridsearch.split_train_dev_test(split_label, ite - 1)
        s_train, s_dev, s_test = gridsearch.split_train_dev_test(split_src, ite - 1)
        t_train, t_dev, t_test = gridsearch.split_train_dev_test(split_trg, ite - 1)
        c_train, c_dev, c_test = gridsearch.split_train_dev_test(split_correct_label, ite - 1)
        ci_train, ci_dev, ci_test = gridsearch.split_train_dev_test(split_correct_index, ite - 1)

        train_iter = dataset.Iterator(s_train, l_train, t_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=True)
        # train_iter = dataset.Iterator(s_train, l_train, t_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=False)
        dev_iter = dataset.Iterator(s_dev, l_dev, t_dev, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        test_iter = dataset.Iterator(s_test, l_test, t_test, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

        logger.info('V{} ## QA data: {}, train size: {}, dev size: {},test size: {}'.format(ite, len(q_train), len(t_train), len(t_dev), len(t_test)))
        mix_train_iter = dataset.MixIterator(qa_iter, train_iter, shuffle=False, multiple=multiple)

        """MODEL"""
        model = model_supervise.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW, trg_initialW)

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
            for i, batch in enumerate(mix_train_iter.generate(), start=1):
                try:
                    if batch[1] == 1.0:
                        loss = optimizer.target(*batch[0])
                    else:
                        loss = model.pretrain(*batch[0], batch[1])
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
                        for b in batch[0][0]:
                            [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]
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
                        for b in batch[0][0]:
                            [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]

                for l in label:
                    labels.append(chainer.cuda.to_cpu(l))

            best_param_dic = evaluater.param_search(labels, alignments, c_dev)

            k = max(best_param_dic, key=lambda x: best_param_dic[x])
            v = best_param_dic[k]
            logger.info('V{} ## E{} ## train loss: {}, dev tuning: {}, {}'.format(ite, epoch, train_loss, k, v))

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
                    with open(model_dir + 'error_log.txt', 'a')as f:
                        f.write('V{} ## E{} ## test iter: {}\n'.format(ite, epoch, i))
                        f.write(traceback.format_exc())
                        f.write('V{} ## E{} ## [batch detail]'.format(ite, epoch))
                        for b in batch[0]:
                            [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]

                for o in output:
                    outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
                for l in label:
                    labels.append(chainer.cuda.to_cpu(l))
                for a in align:
                    alignments.append(chainer.cuda.to_cpu(a))

            init, mix = evaluate.key_to_param(k)
            if model_type in ['multi', 'label', 'pretrain']:
                s_rate, s_count, m_rate, m_count, s_result = evaluater.eval_param(labels, alignments, c_test, ci_test, init, mix)
            else:
                s_rate, s_count, m_rate, m_count, s_result = evaluater.eval_param(alignments, [], c_test, ci_test, init, mix)
            s_result_dic[epoch] = s_result
            logger.info('V{} ## E{} ## {}'.format(ite, epoch, ' '.join(s_rate)))
            # logger.info('V{} ## E{} ## {}'.format(ite, epoch, ' '.join(s_count)))

            dataset.save_output(model_valid_dir, epoch, labels, alignments, outputs)
            accuracy_dic[epoch] = [float(v), s_rate]

        """MODEL SAVE"""
        best_epoch = max(accuracy_dic, key=(lambda x: accuracy_dic[x][0]))
        s_result_total.extend(s_result_dic[best_epoch])
        cross_valid_result.append([ite, best_epoch, accuracy_dic[best_epoch][1]])
        logger.info('V{} ## best_epoch:{} {}'.format(ite, best_epoch, model_valid_dir))
        chainer.serializers.save_npz(model_valid_dir + 'best_model.npz', model)
        dataset.copy_best_output(model_valid_dir, best_epoch)

        logger.info('')

    average_score = [0, 0, 0, 0, 0, 0, 0]
    for r in cross_valid_result:
        average_score = [average_score[i] + float(r[2][i]) for i in range(len(average_score))]
        logger.info('{}: epoch{}, {}'.format(r[0], r[1], ' '.join(r[2])))
    average_score = [str(average_score[i] / len(cross_valid_result)) for i in range(len(average_score))]
    logger.info('average: {}'.format(' '.join(average_score)))

    with open(model_dir + 's_res.txt', 'w')as f:
        [f.write('{},{}\n'.format(l[0], l[1])) for l in sorted(s_result_total, key=lambda x: x[0])]


if __name__ == '__main__':
    main()