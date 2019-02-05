import copy
import pickle
import os
import random
import numpy as np
import logging
from logging import getLogger
from collections import Counter, defaultdict

import convert


def load_pickle(file_name):
    with open(file_name, 'rb')as f:
        data = pickle.load(f)
    return data


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return


def save_output(save_dir, epoch_info):
    epoch = epoch_info['epoch']
    remove_lit = ['id', 'label', 'align', 'hypo', 'tf']

    if epoch_info['label']:
        with open(save_dir + 'model_epoch_{}.label'.format(epoch), 'w')as f:
            [f.write('{}\n'.format(l)) for l in epoch_info['label']]
    if epoch_info['align']:
        with open(save_dir + 'model_epoch_{}.align'.format(epoch), 'w')as f:
            [f.write('{}\n'.format(a)) for a in epoch_info['align']]
    if epoch_info['hypo']:
        with open(save_dir + 'model_epoch_{}.hypo'.format(epoch), 'w')as f:
            [f.write('{}\n'.format(h)) for h in epoch_info['hypo']]
    with open(save_dir + 'model_epoch_{}.tf.txt'.format(epoch), 'w')as f:
        [f.write('{}\n'.format(l)) for l in epoch_info['tf']]
    with open(save_dir + 'model_epoch_{}.score'.format(epoch), 'w')as f:
        for k, v in sorted(epoch_info.items()):
            if k in remove_lit:
                continue
            f.write('{}: {}\n'.format(k, v))
    return


def save_list(file_name, lit):
    if lit:
        with open(file_name, 'w')as f:
            [f.write('{}\n'.format(l)) for l in lit]
    return


def sort_multi_list(id, label, align, tf):
    if label:
        if align:
            c = list(zip(id, label, align, tf))
            c.sort()
            id, label, align, tf = zip(*c)
            return label, align, tf
        else:
            c = list(zip(id, label, tf))
            c.sort()
            id, label, tf = zip(*c)
            return label, [], tf
    else:
        c = list(zip(id, align, tf))
        c.sort()
        id, align, tf = zip(*c)
        return [], align, tf


def load_score_file(model_name):
    label = []
    align = []

    def load_score_file(score_file):
        score_label = []
        with open(score_file, 'r')as f:
            data = f.readlines()
        for line in data:
            line = line[1:-2]
            score = np.array([float(l) for l in line.split()])
            score_label.append(score)
        return score_label

    label_file = model_name + '.label'
    if os.path.isfile(label_file):
        label = load_score_file(label_file)

    align_file = model_name + '.align'
    if os.path.isfile(align_file):
        align = load_score_file(align_file)

    return label, align


def load_label_corpus_file(src_file_name, trg_file_name, separator='|||'):
    data_lit = []

    with open(src_file_name, 'r')as f:
        s_data = f.readlines()
    with open(trg_file_name, 'r')as f:
        t_data = f.readlines()
    for index, (src, trg) in enumerate(zip(s_data, t_data), start=1):
        dic = {}
        label, sentences = src.strip().split('\t')
        sentences = sentences.split(separator)

        dic['id'] = index
        dic['sent_num'] = len(sentences)
        dic['label'] = int(label) - 1
        binary_label_lit = np.array([1 if i == dic['label'] else 0 for i in range(len(sentences))], dtype=np.int32)
        dic['b_label_lit'] = binary_label_lit

        text = []
        for sentence in sentences:
            text.append(sentence.split(' '))
        dic['text'] = text

        dic['ans'] = trg.strip().split(' ')

        data_lit.append(dic)

    return data_lit


def split_valid_data(data, valid=5, debug=False):
    class_data = defaultdict(list)
    data_sub_lit = [[] for _ in range(valid)]

    for d in data:
        if type(d) == dict:
            sent_num = d['sent_num'] if d['sent_num'] <= 7 else 7
        else:
            sent_num = len(d) if len(d) <= 7 else 7
        class_data[sent_num].append(d)

    for k, v in class_data.items():
        group_size = int(len(v) / valid)
        start = 0
        end = group_size
        for i in range(valid):
            data_sub_lit[i].extend(v[start:end])

            start = end
            end += group_size
        else:
            data_sub_lit[i].extend(v[start:])

    if debug:
        print('sub_list:', len(data_sub_lit))
        for i, d in enumerate(data_sub_lit, start=1):
            count = []
            for dd in d:
                count.append(dd['sent_num'])
            print('{}: {}'.format(i, len(d)))
            print(Counter(count))

    return data_sub_lit


def float_to_str(lit):
    return ' '.join([str(round(l, 3)) for l in lit])


def separate_train_dev_test(data, index):
    train = []
    dev = []
    test = []

    valid_num = len(data)

    dev_index = index + valid_num - 2
    if dev_index > valid_num:
        dev_index -= valid_num

    test_index = index - 1
    if index == 1:
        test_index = valid_num

    for i, d in enumerate(data, start=1):
        if i == dev_index:
            dev.extend(d)
        elif i == test_index:
            test.extend(d)
        else:
            train.extend(d)
    return train, dev, test


def prepare_logger(log_file):
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def prepare_vocab(model_dir, data, vocab_size, gpu_id):
    src_vocab = VocabNormal()
    trg_vocab = VocabNormal()
    if os.path.isfile(model_dir + 'src_vocab.normal.pkl') and os.path.isfile(model_dir + 'trg_vocab.normal.pkl'):
        src_vocab.load(model_dir + 'src_vocab.normal.pkl')
        trg_vocab.load(model_dir + 'trg_vocab.normal.pkl')
    else:
        init_vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        src_vocab.build(data, init_vocab, vocab_size, type='src')
        trg_vocab.build(data, init_vocab, vocab_size, type='trg')
        save_pickle(model_dir + 'src_vocab.normal.pkl', src_vocab.vocab)
        save_pickle(model_dir + 'trg_vocab.normal.pkl', trg_vocab.vocab)
    src_vocab.set_reverse_vocab()
    trg_vocab.set_reverse_vocab()

    sos = convert.convert_list(np.array([src_vocab.vocab['<s>']], dtype=np.int32), gpu_id)
    eos = convert.convert_list(np.array([src_vocab.vocab['</s>']], dtype=np.int32), gpu_id)

    return src_vocab, trg_vocab, sos, eos


class VocabNormal:
    def __init__(self):
        self.vocab = None
        self.reverse_vocab = None

    def build(self, data, init_vocab, vocab_size, type):
        vocab = copy.copy(init_vocab)
        word_count = Counter()
        words = []

        if type == 'src':
            for i, d in enumerate(data, start=1):
                for sentence in d['text']:
                    words.extend(sentence)
                # 10000文書ごとにCounterへ渡す
                if i % 10000 == 0:
                    word_count += Counter(words)
                    words = []
            else:
                word_count += Counter(words)

        elif type == 'trg':
            for i, d in enumerate(data, start=1):
                words.extend(d['ans'])
                # 10000文書ごとにCounterへ渡す
                if i % 10000 == 0:
                    word_count += Counter(words)
                    words = []
            else:
                word_count += Counter(words)

        for w, c in word_count.most_common():
            if len(vocab) >= vocab_size:
                break
            if w not in vocab:
                vocab[w] = len(vocab)
        self.vocab = vocab
        return

    def load(self, vocab_file):
        self.vocab = load_pickle(vocab_file)

    def word2id(self, sentence, sos=False, eos=False):
        vocab = self.vocab
        sentence_id = [vocab.get(word, self.vocab['<unk>']) for word in sentence]

        if sos:
            sentence_id.insert(0, self.vocab['<s>'])
        if eos:
            sentence_id.append(self.vocab['</s>'])

        return np.array(sentence_id, dtype=np.int32)

    def set_reverse_vocab(self):
        reverse_vocab = {}
        for k, v in self.vocab.items():
            reverse_vocab[v] = k
        self.reverse_vocab = reverse_vocab

    def id2word(self, sentence_id):
        sentence = [self.reverse_vocab.get(word, '<unk>') for word in sentence_id]
        sentence = ' '.join(sentence)
        return sentence


class Iterator:
    def __init__(self, data, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.size = len(data)

        self.sort = sort
        self.shuffle = shuffle

        self.batches = self._prepare_minibatch(data, batch_size, gpu_id)

    def _convert(self, data):
        src_id = [self.src_vocab.word2id(t) for t in data['text']]
        trg_sos = self.trg_vocab.word2id(data['ans'], sos=True)
        trg_eos = self.trg_vocab.word2id(data['ans'], eos=True)
        label = data['b_label_lit']
        return src_id, trg_sos, trg_eos, label

    def _prepare_minibatch(self, data, batch_size, gpu_id):
        batch_data = []
        for d in data:
            batch_data.append(self._convert(d))

        if self.sort:
            batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)
        batches = [convert.convert(batch_data[b * batch_size: (b + 1) * batch_size], gpu_id) for b in range(len(batch_data) // batch_size)]
        if len(batch_data) % batch_size != 0:
            batches.append(convert.convert(batch_data[-(len(batch_data) % batch_size):], gpu_id))

        return batches

    def generate(self):
        batches = self.batches
        if self.shuffle:
            batches = random.sample(batches, len(batches))

        for batch in batches:
            yield batch


class MixIterator:
    def __init__(self, iterator1, iterator2, seed, shuffle=True, type='over', multiple=1):
        # iterator1を大きいデータサイズのiteratorに指定する
        self.batches = []

        if type == 'over':
            for batch in iterator1.batches:
                self.batches.append([batch, False])
            for i in range(multiple):
                for batch in iterator2.batches:
                    self.batches.append([batch, True])

        elif type == 'under':
            random.seed(seed)
            ite = random.sample(iterator1.batches, iterator2.size * multiple)
            for batch in ite:
                self.batches.append([batch, False])
            for i in range(multiple):
                for batch in iterator2.batches:
                    self.batches.append([batch, True])

        self.shuffle = shuffle

    def generate(self):
        batches = self.batches
        if self.shuffle:
            batches = random.sample(batches, len(batches))

        for batch in batches:
            yield batch[0], batch[1]