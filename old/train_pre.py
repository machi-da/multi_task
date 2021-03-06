import argparse
import configparser
import os
import shutil
import numpy as np

from old import evaluate, dataset, gridsearch
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
    parser.add_argument('--load_model', '-l', action='store_true')
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
    load_model = args.load_model

    """DIR PREPARE"""
    config.read(config_file)
    vocab_size = int(config['Parameter']['vocab_size'])
    coefficient = float(config['Parameter']['coefficient'])

    vocab_name = vocab_type
    if pretrain_w2v:
        vocab_name = 'p' + vocab_name

    model_dir = './pre_{}_{}{}_{}/'.format(model_type, vocab_name, vocab_size, data_path[0])

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

    train_data_size = dataset.data_size(train_src_file)
    valid_data_size = dataset.data_size(valid_src_file)

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

    """MODEL"""
    if model_type == 'multi':
        model = model.Multi(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, coefficient, src_initialW, trg_initialW)
    elif model_type in ['label', 'pretrain']:
        model = model.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW, trg_initialW)
    else:
        model = model.EncoderDecoder(src_vocab_size, trg_vocab_size, embed_size, hidden_size, dropout_ratio, src_initialW, trg_initialW)
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
    if not load_model:
        logger.info('Pre-train start')
        logger.info('train size: {}, valid size: {}'.format(train_data_size, valid_data_size))
        _, src_label, src_text, _ = dataset.load_binary_score_file(train_src_file)
        trg_text = dataset.load(train_trg_file)
        train_iter = dataset.Iterator(src_text, src_label, trg_text, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True)
        # train_iter = dataset.Iterator(src_text, src_label, trg_text, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

        _, src_label, src_text, _ = dataset.load_binary_score_file(valid_src_file)
        trg_text = dataset.load(valid_trg_file)
        valid_iter = dataset.Iterator(src_text, src_label, trg_text, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

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
                    # with open(model_dir + 'error_log.txt', 'a')as f:
                    #     f.write('P{} ## iteration {}\n'.format(epoch, i))
                    #     f.write(traceback.format_exc())
                    #     for b in batch[0]:
                    #         [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]
            chainer.serializers.save_npz(model_dir + 'p_model_epoch_{}.npz'.format(epoch), model)

            """EVALUATE"""
            valid_loss = 0
            for batch in valid_iter.generate():
                with chainer.no_backprop_mode(), chainer.using_config('train', False):
                    valid_loss += model.pretrain(*batch).data
            logger.info('P{} ## train loss: {}, val loss:{}'.format(epoch, train_loss, valid_loss))
            pretrain_loss_dic[epoch] = valid_loss

        """MODEL SAVE"""
        best_epoch = min(pretrain_loss_dic, key=(lambda x: pretrain_loss_dic[x]))
        logger.info('best_epoch:{}, val loss: {}'.format(best_epoch, pretrain_loss_dic[best_epoch]))
        shutil.copyfile(model_dir + 'p_model_epoch_{}.npz'.format(best_epoch), model_dir + 'p_best_model.npz')
        logger.info('Pre-train finish')

    """MAIN"""
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

        train_iter = dataset.Iterator(src_train, label_train, trg_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=True)
        # train_iter = dataset.Iterator(src_train, label_train, trg_train, src_vocab, trg_vocab, batch_size, gpu_id, sort=True)
        dev_iter = dataset.Iterator(src_dev, label_dev, trg_dev, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)
        test_iter = dataset.Iterator(src_test, label_test, trg_test, src_vocab, trg_vocab, batch_size, gpu_id, sort=False, shuffle=False)

        logger.info('V{} ## train: {}, dev: {},test: {}'.format(ite, len(label_train), len(label_dev), len(label_test)))

        """MODEL"""
        model = model.Label(src_vocab_size, trg_vocab_size, embed_size, hidden_size, class_size, dropout_ratio, src_initialW, trg_initialW)
        chainer.serializers.load_npz(model_dir + 'p_best_model.npz', model)
        if gpu_id >= 0:
            model.to_gpu()

        """OPTIMIZER"""
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

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
                    logger.info('V{} ## E{} ## train iter: {}, {}'.format(ite, epoch, i, e))
                    # with open(model_dir + 'error_log.txt', 'a')as f:
                    #     f.write('V{} ## E{} ## train iter: {}\n'.format(ite, epoch, i))
                    #     f.write(traceback.format_exc())
                    #     f.write('V{} ## E{} ## [batch detail]'.format(ite, epoch))
                    #     for b in batch[0]:
                    #         [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]
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
                    # with open(model_dir + 'error_log.txt', 'a')as f:
                    #     f.write('V{} ## E{} ## dev iter: {}\n'.format(ite, epoch, i))
                    #     f.write(traceback.format_exc())
                    #     f.write('V{} ## E{} ## [batch detail]'.format(ite, epoch))
                    #     for b in batch[0]:
                    #         [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]

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
                    # with open(model_dir + 'error_log.txt', 'a')as f:
                    #     f.write('V{} ## E{} ## test iter: {}\n'.format(ite, epoch, i))
                    #     f.write(traceback.format_exc())
                    #     f.write('V{} ## E{} ## [batch detail]'.format(ite, epoch))
                    #     for b in batch[0]:
                    #         [f.write(src_vocab.id2word(chainer.cuda.to_cpu(bb)) + '\n') for bb in b]

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
            logger.info('V{} ## E{} ## loss:{}, dev: {}, test: {}'.format(ite, epoch, train_loss, dev_score, test_score))

            dataset.save_output(model_valid_dir, epoch, labels, alignments, outputs, s_result)
            accuracy_dic[epoch] = [epoch, dev_score, test_score, param, s_rate, s_result]

        """MODEL SAVE"""
        best_epoch = max(accuracy_dic, key=(lambda x: accuracy_dic[x][1]))
        cross_valid_result.append(accuracy_dic[best_epoch])
        logger.info('V{} ## best_epoch:{}, dev:{}, test:{}'.format(ite, best_epoch, accuracy_dic[best_epoch][1], accuracy_dic[best_epoch][2]))
        shutil.copyfile(model_valid_dir + 'model_epoch_{}.npz'.format(best_epoch), model_valid_dir + 'best_model.npz')

        logger.info('')

    average_dev_score = 0
    average_test_score = [0 for _ in range(len(cross_valid_result[0][4]))]
    s_result_total = []
    for i, r in enumerate(cross_valid_result, start=1):
        epoch = r[0]
        dev_score = r[1]
        param = r[3]
        test_score_list = [round(rr, 3) for rr in r[4]]
        s_result = r[5]

        average_dev_score += dev_score
        average_test_score = [average_test_score[i] + test_score_list[i] for i in range(len(average_test_score))]
        logger.info('   {}: epoch{}, {}\t{}'.format(i, epoch, param, ' '.join(dataset.float_to_str(test_score_list))))
        s_result_total.extend(s_result)
    average_dev_score = round(average_dev_score / len(cross_valid_result), 3)
    average_test_score = [round(average_test_score[i] / len(cross_valid_result), 3) for i in range(len(average_test_score))]
    logger.info('dev: {}, test: {}'.format(average_dev_score, ' '.join(dataset.float_to_str(average_test_score))))

    with open(model_dir + 's_res.txt', 'w')as f:
        [f.write('{}\n'.format(l[1])) for l in sorted(s_result_total, key=lambda x: x[0])]


if __name__ == '__main__':
    main()