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
    parser.add_argument('--batch', '-b', type=int, default=20)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--pretrain_epoch', '-pe', type=int, default=10)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--model', '-m', choices=['multi', 'label', 'encdec', 'pretrain'], default='multi')
    parser.add_argument('--pretrain_w2v', '-p', action='store_true')
    parser.add_argument('--data_path', '-d', choices=['local', 'server', 'test'], default='server')
    parser.add_argument('--load_model', '-l', type=str)
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
    pretrain_w2v = args.pretrain_w2v
    data_path = args.data_path
    load_model = args.load_model

    """DIR PREPARE"""
    config.read(config_file)
    vocab_size = int(config['Parameter']['vocab_size'])
    coefficient = float(config['Parameter']['coefficient'])

    if pretrain_w2v:
        vocab_size = 'p' + str(vocab_size)

    if model_type == 'multi':
        base_dir = './{}_{}_{}_c{}/'.format(model_type, vocab_size, data_path[0], coefficient)
    else:
        base_dir = './{}_{}_{}/'.format(model_type, vocab_size, data_path[0])
    model_save_dir = base_dir

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        shutil.copyfile(config_file, base_dir + config_file)
    config_file = base_dir + config_file
    config.read(config_file)

    if load_model is not None:
        model_save_dir = base_dir + load_model.replace('.npz', '') + '/'
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

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
    log_file = model_save_dir + 'log.txt'
    logger = dataset_new.prepare_logger(log_file)

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

    train_data = dataset_new.load_label_corpus_file(train_src_file, train_trg_file)
    valid_data = dataset_new.load_label_corpus_file(valid_src_file, valid_trg_file)
    test_data = dataset_new.load_label_corpus_file(test_src_file, test_trg_file)

    logger.info('train size:{}, valid size:{}, test size:{}'.format(len(train_data), len(valid_data), len(test_data)))

    """VOCABULARY"""
    src_vocab, trg_vocab, sos, eos = dataset_new.prepare_vocab(base_dir, train_data, vocab_size, gpu_id)
    src_vocab_size = len(src_vocab.vocab)
    trg_vocab_size = len(trg_vocab.vocab)

    src_initialW, trg_initialW = None, None
    if pretrain_w2v:
        w2v = word2vec.Word2Vec()
        src_initialW, vector_size, src_match_word_count = w2v.make_initialW(src_vocab.vocab, src_w2v_file)
        trg_initialW, vector_size, trg_match_word_count = w2v.make_initialW(trg_vocab.vocab, trg_w2v_file)
        logger.info('Initialize w2v embedding. Match: src {}/{}, trg {}/{}'.format(src_match_word_count, src_vocab_size, trg_match_word_count, trg_vocab_size))

    logger.info('src_vocab size: {}, trg_vocab size: {}'.format(src_vocab_size, trg_vocab_size))

    """ITERATOR"""
    train_iter = dataset_new.Iterator(train_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True)
    # train_iter = dataset_new.Iterator(train_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
    valid_iter = dataset_new.Iterator(valid_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

    test_sub_lit = dataset_new.split_valid_data(test_data, valid_num)
    dev_test_iter_lit = []
    for ite in range(1, valid_num + 1):
        _, dev_data, test_data = dataset_new.separate_train_dev_test(test_sub_lit, ite)
        dev_iter = dataset_new.Iterator(dev_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        test_iter = dataset_new.Iterator(test_data, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        id_lit = []
        for t in test_data:
            id_lit.append(t['id'])
        dev_test_iter_lit.append([dev_iter, test_iter, id_lit, test_data])

    """MODEL"""
    if model_type == 'multi':
        model = model_supervise.Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, src_initialW, trg_initialW)
    elif model_type in ['label', 'pretrain']:
        model = model_supervise.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW, trg_initialW)
    else:
        model = model_supervise.EncoderDecoder(src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_ratio, src_initialW, trg_initialW)

    evaluater = evaluate_new.Evaluate()

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
    if model_type == 'pretrain' and load_model is None:
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
                    logger.info('P{} ## train iter: {}, {}'.format(epoch, i, e))
            chainer.serializers.save_npz(model_save_dir + 'p_model_epoch_{}.npz'.format(epoch), model)

            """EVALUATE"""
            valid_loss = 0
            for batch in valid_iter.generate():
                with chainer.no_backprop_mode(), chainer.using_config('train', False):
                    valid_loss += model.pretrain(*batch).data
            logger.info('P{} ## train loss: {}, val loss:{}'.format(epoch, train_loss, valid_loss))
            pretrain_loss_dic[epoch] = valid_loss

        """MODEL SAVE & LOAD"""
        best_epoch = min(pretrain_loss_dic, key=(lambda x: pretrain_loss_dic[x]))
        logger.info('best_epoch:{}, val loss: {}'.format(best_epoch, pretrain_loss_dic[best_epoch]))
        shutil.copyfile(model_save_dir + 'p_model_epoch_{}.npz'.format(best_epoch), model_save_dir + 'p_best_model.npz')
        logger.info('Pre-train finish')

    if load_model:
        logger.info('load model: {}'.format(load_model))
        chainer.serializers.load_npz(model_save_dir + load_model, model)

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

            except Exception as e:
                logger.info('E{} ## train iter: {}, {}'.format(epoch, i, e))

        chainer.serializers.save_npz(model_save_dir + 'model_epoch_{}.npz'.format(epoch), model)

        """DEV & TEST"""
        dev_test_info = {}
        for ite, lit in enumerate(dev_test_iter_lit, start=1):
            dev_iter = lit[0]
            test_iter = lit[1]
            test_id = lit[2]
            test_data = lit[3]
            """DEV"""
            outputs, labels, alignments = [], [], []
            for i, batch in enumerate(dev_iter.generate(), start=1):
                try:
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        output, label, align = model.predict(batch[0], sos, eos)
                except Exception as e:
                    logger.info('E{} ## test iter: {}, {}'.format(epoch, i, e))

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

            if model_type == 'encdec':
                best_param_dic = evaluater.param_search(alignments, [], dev_data)
            else:
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
                rate, count, tf_lit, macro, micro = evaluater.eval_param(labels, alignments, test_data, init, mix)
            else:
                rate, count, tf_lit, macro, micro = evaluater.eval_param(alignments, [], test_data, init, mix)
            test_macro_score = round(macro, 3)
            test_micro_score = round(micro, 3)
            logger.info('E{} ##   {}: {}\tdev:{}, micro:{}, macro:{} {}'.format(epoch, ite, param, dev_score, test_micro_score, dataset_new.float_to_str(rate), test_macro_score))

            dev_test_info[ite] = {
                'id': test_id,
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

        ave_dev_score = 0
        ave_macro_score = 0
        ave_micro_score = 0
        ave_test_score = [0 for _ in range(len(dev_test_info[1]['rate']))]
        id_total = []
        label_total = []
        align_total = []
        tf_total = []
        for k, v in dev_test_info.items():
            ave_dev_score += v['dev_score']
            ave_macro_score += v['macro']
            ave_micro_score += v['micro']
            for i, rate in enumerate(v['rate']):
                ave_test_score[i] += rate
            id_total.extend(v['id'])
            label_total.extend(v['label'])
            align_total.extend(v['align'])
            tf_total.extend(v['tf'])
        ave_dev_score = round(ave_dev_score / valid_num, 3)
        ave_macro_score = round(ave_macro_score / valid_num, 3)
        ave_micro_score = round(ave_micro_score / valid_num, 3)
        ave_test_score = [ave_test_score[i] / valid_num for i in range(len(ave_test_score))]
        logger.info('E{} ## loss:{}, dev:{}, micro:{}, macro:{} {}'.format(epoch, train_loss, ave_dev_score, ave_micro_score, dataset_new.float_to_str(ave_test_score), ave_macro_score))

        label, align, tf = dataset_new.sort_multi_list(id_total, label_total, align_total, tf_total)

        if label:
            with open(model_save_dir + 'model_epoch_{}.label'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(l)) for l in label]
        if align:
            with open(model_save_dir + 'model_epoch_{}.align'.format(epoch), 'w')as f:
                [f.write('{}\n'.format(a)) for a in align]
        with open(model_save_dir + 'model_epoch_{}.tf'.format(epoch), 'w')as f:
            [f.write('{}\n'.format(l)) for l in tf]

        accuracy_dic[epoch] = [ave_dev_score, ave_micro_score, ave_macro_score]

    """MODEL SAVE"""
    best_epoch = max(accuracy_dic, key=(lambda x: accuracy_dic[x][0]))
    logger.info('best_epoch:{}, dev:{}, micro:{}, macro:{}'.format(best_epoch, accuracy_dic[best_epoch][0], accuracy_dic[best_epoch][1], accuracy_dic[best_epoch][2], model_save_dir))
    shutil.copyfile(model_save_dir + 'model_epoch_{}.npz'.format(best_epoch), model_save_dir + 'best_model.npz')


if __name__ == '__main__':
    main()