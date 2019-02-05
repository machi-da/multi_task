import argparse
import configparser
import os
import shutil
import numpy as np
import traceback

import dataset_new
import evaluate_new
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
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', choices=['multi'], default='multi')
    parser.add_argument('--pretrain_w2v', '-p', action='store_true')
    parser.add_argument('--data_path', '-d', choices=['local', 'server', 'test'], default='server')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = configparser.ConfigParser()
    """ARGS DETAIL"""
    config_file = args.config_file
    batch_size = args.batch
    n_epoch = args.epoch
    gpu_id = args.gpu
    model_type = args.model
    pretrain_w2v = args.pretrain_w2v
    data_path = args.data_path

    """DIR PREPARE"""
    config.read(config_file)
    vocab_size = int(config['Parameter']['vocab_size'])
    coefficient = float(config['Parameter']['coefficient'])
    sample_type = config['Parameter']['sample_type']
    multiple = int(config['Parameter']['multiple'])
    shuffle = bool(int(config['Parameter']['shuffle']))

    if pretrain_w2v:
        vocab_size = 'p' + str(vocab_size)

    if multiple == 1:
        if shuffle:
            model_dir = './mix_{}_{}_{}_c{}_{}_shuffle/'.format(model_type, vocab_size, data_path[0], coefficient, sample_type)
        else:
            model_dir = './mix_{}_{}_{}_c{}_{}/'.format(model_type, vocab_size, data_path[0], coefficient, sample_type)
    else:
        if shuffle:
            model_dir = './mix_{}_{}_{}_c{}_{}{}_shuffle/'.format(model_type, vocab_size, data_path[0], coefficient, sample_type, multiple)
        else:
            model_dir = './mix_{}_{}_{}_c{}_{}{}/'.format(model_type, vocab_size, data_path[0], coefficient, sample_type, multiple)

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
    valid_num = int(config['Parameter']['valid_num'])
    sample_type = config['Parameter']['sample_type']
    multiple = int(config['Parameter']['multiple'])
    shuffle = bool(int(config['Parameter']['shuffle']))
    """LOGGER"""
    log_file = model_dir + 'log.txt'
    logger = dataset_new.prepare_logger(log_file)
    logger.info(args)  # 引数を記録
    logger.info('[Training start] logging to {}'.format(log_file))

    """DATASET"""
    train_src_file = config[data_path]['train_src_file']
    train_trg_file = config[data_path]['train_trg_file']
    test_src_file = config[data_path]['single_src_file']
    test_trg_file = config[data_path]['single_trg_file']
    src_w2v_file = config[data_path]['src_w2v_file']
    trg_w2v_file = config[data_path]['trg_w2v_file']

    train_data = dataset_new.load_label_corpus_file(train_src_file, train_trg_file)
    qa_data_sub_lit = dataset_new.split_valid_data(train_data, valid_num)
    test_data = dataset_new.load_label_corpus_file(test_src_file, test_trg_file)
    test_data_sub_lit = dataset_new.split_valid_data(test_data, valid_num)

    """VOCABULARY"""
    src_vocab, trg_vocab, sos, eos = dataset_new.prepare_vocab(model_dir, train_data, vocab_size, gpu_id)
    src_vocab_size = len(src_vocab.vocab)
    trg_vocab_size = len(trg_vocab.vocab)

    src_initialW, trg_initialW = None, None
    if pretrain_w2v:
        w2v = word2vec.Word2Vec()
        src_initialW, vector_size, src_match_word_count = w2v.make_initialW(src_vocab.vocab, src_w2v_file)
        trg_initialW, vector_size, trg_match_word_count = w2v.make_initialW(trg_vocab.vocab, trg_w2v_file)
        logger.info('Initialize w2v embedding. Match: src {}/{}, trg {}/{}'.format(src_match_word_count, src_vocab_size, trg_match_word_count, trg_vocab_size))

    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    evaluater = evaluate_new.Evaluate()

    """GPU"""
    if gpu_id >= 0:
        logger.info('Use GPU')
        chainer.cuda.get_device_from_id(gpu_id).use()

    cross_valid_result = []
    for ite in range(1, valid_num + 1):
        model_valid_dir = model_dir + 'valid{}/'.format(ite)
        if not os.path.exists(model_valid_dir):
            os.mkdir(model_valid_dir)

        qa_train_data, qa_dev_data, qa_test_data = dataset_new.separate_train_dev_test(qa_data_sub_lit, ite)
        train_data, dev_data, test_data = dataset_new.separate_train_dev_test(test_data_sub_lit, ite)
        test_data_id = [t['id'] for t in test_data]

        qa_iter = dataset_new.Iterator(qa_train_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True)
        train_iter = dataset_new.Iterator(train_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True)
        dev_iter = dataset_new.Iterator(dev_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        test_iter = dataset_new.Iterator(test_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

        if sample_type == 'over':
            qa_size = len(qa_train_data)
            train_size = len(train_data) * multiple
        elif sample_type == 'under':
            qa_size = len(train_data) * multiple
            train_size = len(train_data)
        logger.info('V{} ## QA:{}, train:{}, dev:{} ,test:{}'.format(ite, qa_size, train_size, len(dev_data), len(test_data)))

        """MODEL"""
        model = model_supervise.Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, src_initialW, trg_initialW)
        if gpu_id >= 0:
            model.to_gpu()

        """OPTIMIZER"""
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

        """TRAIN"""
        epoch_info = {}
        for epoch in range(1, n_epoch + 1):
            train_loss = 0
            seed = ite * 10 + epoch
            mix_train_iter = dataset_new.MixIterator(qa_iter, train_iter, seed, shuffle=shuffle, type=sample_type, multiple=multiple)
            for i, batch in enumerate(mix_train_iter.generate(), start=1):
                try:
                    # loss = optimizer.target(*batch)
                    if batch[1]:
                        loss = optimizer.target(*batch[0])
                    else:
                        loss = model.pretrain(*batch[0])
                    train_loss += loss.data
                    optimizer.target.cleargrads()
                    loss.backward()
                    optimizer.update()

                except Exception as e:
                    logger.info('V{} ## E{} ## train iter: {}, {}'.format(ite, epoch, i, e))
            chainer.serializers.save_npz(model_valid_dir + 'model_epoch_{}.npz'.format(epoch), model)

            """DEV"""
            labels, alignments = [], []
            for i, batch in enumerate(dev_iter.generate(), start=1):
                try:
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        _, label, align = model.predict(batch[0], sos, eos)
                except Exception as e:
                    logger.info('V{} ## E{} ## dev iter: {}, {}'.format(ite, epoch, i, e))

                for l, a in zip(label, align):
                    labels.append(chainer.cuda.to_cpu(l))
                    alignments.append(chainer.cuda.to_cpu(a))

            best_param_dic = evaluater.param_search(labels, alignments, dev_data)
            param = max(best_param_dic, key=lambda x: best_param_dic[x]['macro'])
            init, mix = evaluate_new.key_to_param(param)
            dev_score = round(best_param_dic[param]['macro'], 3)

            """TEST"""
            outputs, labels, alignments = [], [], []
            for i, batch in enumerate(test_iter.generate(), start=1):
                try:
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        output, label, align = model.predict(batch[0], sos, eos)
                except Exception as e:
                    logger.info('V{} ## E{} ## test iter: {}, {}'.format(ite, epoch, i, e))

                for o, l, a in zip(output, label, align):
                    outputs.append(trg_vocab.id2word(chainer.cuda.to_cpu(o)))
                    labels.append(chainer.cuda.to_cpu(l))
                    alignments.append(chainer.cuda.to_cpu(a))

            rate, count, tf_lit, macro, micro = evaluater.eval_param(labels, alignments, test_data, init, mix)
            test_macro_score = round(macro, 3)
            test_micro_score = round(micro, 3)
            logger.info('V{} ## E{} ## loss: {}, dev: {}, param: {}, micro: {}, macro: {}'.format(ite, epoch, train_loss, dev_score, param, test_micro_score, test_macro_score))

            epoch_info[epoch] = {
                'id': test_data_id,
                'label': labels,
                'align': alignments,
                'hypo': outputs,
                'epoch': epoch,
                'dev_score': dev_score,
                'param': param,
                'rate': rate,
                'count': count,
                'tf': tf_lit,
                'macro': test_macro_score,
                'micro': test_micro_score
            }
            dataset_new.save_output(model_valid_dir, epoch_info[epoch])

        """MODEL SAVE"""
        best_epoch = max(epoch_info, key=(lambda x: epoch_info[x]['dev_score']))
        cross_valid_result.append(epoch_info[best_epoch])
        logger.info('V{} ## best_epoch: {}, dev: {}, micro: {}, macro: {}'.format(ite, best_epoch, epoch_info[best_epoch]['dev_score'], epoch_info[best_epoch]['micro'], epoch_info[best_epoch]['macro']))
        shutil.copyfile(model_valid_dir + 'model_epoch_{}.npz'.format(best_epoch), model_valid_dir + 'best_model.npz')

        logger.info('')

    ave_dev_score, ave_macro_score, ave_micro_score = 0, 0, 0
    ave_test_score = [0 for _ in range(len(cross_valid_result[0]['rate']))]
    id_total, label_total, align_total, tf_total = [], [], [], []

    for v, r in enumerate(cross_valid_result, start=1):
        ave_dev_score += r['dev_score']
        ave_macro_score += r['macro']
        ave_micro_score += r['micro']
        for i, rate in enumerate(r['rate']):
            ave_test_score[i] += rate
        logger.info('   {}: e{}, {}\tdev: {}, micro: {}, macro: {} {}'.format(v, r['epoch'], r['param'], r['dev_score'], r['micro'], dataset_new.float_to_str(r['rate']), r['macro']))

        id_total.extend(r['id'])
        label_total.extend(r['label'])
        align_total.extend(r['align'])
        tf_total.extend(r['tf'])
    ave_dev_score = round(ave_dev_score / valid_num, 3)
    ave_macro_score = round(ave_macro_score / valid_num, 3)
    ave_micro_score = round(ave_micro_score / valid_num, 3)
    ave_test_score = [ave_test_score[i] / valid_num for i in range(len(ave_test_score))]
    logger.info('dev: {}, micro: {}, macro: {} {}'.format(ave_dev_score, ave_micro_score, dataset_new.float_to_str(ave_test_score), ave_macro_score))

    label, align, tf = dataset_new.sort_multi_list(id_total, label_total, align_total, tf_total)
    dataset_new.save_list(model_dir + 'label.txt', label)
    dataset_new.save_list(model_dir + 'align.txt', align)
    dataset_new.save_list(model_dir + 'tf.txt', tf)


if __name__ == '__main__':
    main()