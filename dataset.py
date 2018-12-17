import argparse
import copy
import configparser
import glob
import pickle
import os
import random
import shutil
import numpy as np
import sentencepiece as spm
import logging
from logging import getLogger
from collections import Counter

import convert


def load(file_name):
    text = []
    with open(file_name)as f:
        data = f.readlines()
    for d in data:
        text.append(d.strip().split(' '))
    return text


def load_with_label_binary(file_name):
    label_lit = []
    text = []
    with open(file_name, 'r')as f:
        data = f.readlines()
    for d in data:
        t = []
        label, sentences = d.strip().split('\t')
        sentences = sentences.split('|||')
        l_index = [int(l)-1 for l in label.split(',')]
        label = np.array([1 if i in l_index else 0 for i in range(len(sentences))], dtype=np.int32)
        label_lit.append(label)

        for sentence in sentences:
            t.append(sentence.split(' '))
        text.append(t)
    return label_lit, text


def load_with_label_reg(file_name):
    label_lit = []
    text = []
    with open(file_name, 'r')as f:
        data = f.readlines()
    for d in data:
        t = []
        label, sentences = d.strip().split('\t')
        sentences = sentences.split('|||')
        label = np.array([float(l) for l in label.split(',')], dtype=np.float32)
        label_lit.append(label)

        for sentence in sentences:
            t.append(sentence.split(' '))
        text.append(t)
    return label_lit, text


def load_with_label_index(file_name):
    label_lit = []
    text = []
    index = []

    with open(file_name, 'r')as f:
        data = f.readlines()
    for i, d in enumerate(data, start=1):
        label, sentences = d.strip().split('\t')
        label = [int(l) - 1 for l in label.split(',')]
        sentences = sentences.split('|||')

        label_lit.append(label)
        t = []
        for sentence in sentences:
            t.append(sentence.split(' '))
        text.append(t)
        index.append(i)

    return label_lit, text, index


def load_score_file(score_file):
    score_label = []
    with open(score_file, 'r')as f:
        data = f.readlines()
    for line in data:
        line = line[1:-2]
        score = np.array([float(l) for l in line.split()])
        score_label.append(score)
    return score_label


def data_size(file_name):
    with open(file_name, 'r')as f:
        size = sum([1 for _ in f.readlines()])
    return size


def load_pickle(file_name):
    with open(file_name, 'rb')as f:
        data = pickle.load(f)
    return data


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return


def save_output(save_dir, epoch, label_data, align_data, hypo_data):
    if label_data:
        with open(save_dir + 'model_epoch_{}.label'.format(epoch), 'w')as f:
            [f.write('{}\n'.format(l)) for l in label_data]
    if align_data:
        with open(save_dir + 'model_epoch_{}.hypo'.format(epoch), 'w')as f:
            [f.write(h + '\n') for h in hypo_data]
    if hypo_data:
        with open(save_dir + 'model_epoch_{}.align'.format(epoch), 'w')as f:
            [f.write('{}\n'.format(a)) for a in align_data]
    return


def copy_best_output(save_dir, best_epoch):
    try:
        shutil.copy(save_dir + 'model_epoch_{}.label'.format(best_epoch), save_dir + 'best_model.label')
    except FileNotFoundError:
        pass
    try:
        shutil.copy(save_dir + 'model_epoch_{}.hypo'.format(best_epoch), save_dir + 'best_model.hypo')
    except FileNotFoundError:
        pass
    try:
        shutil.copy(save_dir + 'model_epoch_{}.align'.format(best_epoch), save_dir + 'best_model.aligh')
    except FileNotFoundError:
        pass
    return


def prepare_vocab(model_dir, vocab_type, src_file, trg_file, vocab_size, gpu_id):
    if vocab_type == 'normal':
        src_vocab = VocabNormal()
        trg_vocab = VocabNormal()
        if os.path.isfile(model_dir + 'src_vocab.normal.pkl') and os.path.isfile(model_dir + 'trg_vocab.normal.pkl'):
            src_vocab.load(model_dir + 'src_vocab.normal.pkl')
            trg_vocab.load(model_dir + 'trg_vocab.normal.pkl')
        else:
            init_vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
            src_vocab.build(src_file, True,  init_vocab, vocab_size)
            trg_vocab.build(trg_file, False, init_vocab, vocab_size)
            save_pickle(model_dir + 'src_vocab.normal.pkl', src_vocab.vocab)
            save_pickle(model_dir + 'trg_vocab.normal.pkl', trg_vocab.vocab)
        src_vocab.set_reverse_vocab()
        trg_vocab.set_reverse_vocab()

        sos = convert.convert_list(np.array([src_vocab.vocab['<s>']], dtype=np.int32), gpu_id)
        eos = convert.convert_list(np.array([src_vocab.vocab['</s>']], dtype=np.int32), gpu_id)

    elif vocab_type == 'subword':
        src_vocab = VocabSubword()
        trg_vocab = VocabSubword()
        if os.path.isfile(model_dir + 'src_vocab.sub.model') and os.path.isfile(model_dir + 'trg_vocab.sub.model'):
            src_vocab.load(model_dir + 'src_vocab.sub.model')
            trg_vocab.load(model_dir + 'trg_vocab.sub.model')
        else:
            src_vocab.build(src_file + '.sub', model_dir + 'src_vocab.sub', vocab_size)
            trg_vocab.build(trg_file + '.sub', model_dir + 'trg_vocab.sub', vocab_size)

        sos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('<s>')], dtype=np.int32), gpu_id)
        eos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('</s>')], dtype=np.int32), gpu_id)

    return src_vocab, trg_vocab, sos, eos


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


class VocabNormal:
    def __init__(self):
        self.vocab = None
        self.reverse_vocab = None

    def build(self, obj, with_label, initial_vocab, vocab_size, freq=0):
        vocab = copy.copy(initial_vocab)
        word_count = Counter()
        words = []

        if type(obj) is str:
            if with_label:
                _, documents = load_with_label_reg(obj)

                for i, doc in enumerate(documents):
                    for sentence in doc:
                        words.extend(sentence)
                    # 10000文書ごとにCounterへ渡す
                    if i % 10000 == 0:
                        word_count += Counter(words)
                        words = []
                else:
                    word_count += Counter(words)

            else:
                documents = load(obj)
                for doc in documents:
                    words.extend(doc)
                word_count = Counter(words)

        elif type(obj) is list:
            for i, doc in enumerate(obj):
                for sentence in doc:
                    words.extend(sentence)
                if i % 10000 == 0:
                    word_count += Counter(words)
                    words = []
            else:
                word_count += Counter(words)

        for w, c in word_count.most_common():
            if len(vocab) >= vocab_size:
                break
            if c <= freq:
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


class VocabSubword:
    def __init__(self):
        self.vocab = None

    def build(self, file_name, model_name, vocab_size):
        self.vocab = self._build_vocab(file_name, model_name, vocab_size)

    def _build_vocab(self, text_file, model_name, vocab_size):
        args = '''
                --pad_id=0 
                --unk_id=1 
                --bos_id=2
                --eos_id=3
                --input={} 
                --model_prefix={} 
                --vocab_size={} 
                --hard_vocab_limit=false''' \
            .format(text_file, model_name, vocab_size)
        spm.SentencePieceTrainer.Train(args)
        sp = spm.SentencePieceProcessor()
        sp.Load(model_name + '.model')
        return sp

    def load(self, vocab_file):
        sp = spm.SentencePieceProcessor()
        sp.Load(vocab_file)
        self.vocab = sp

    def word2id(self, sentence, sos=False, eos=False):
        sp = self.vocab
        sentence_id = sp.EncodeAsIds(' '.join(sentence))

        if sos:
            sentence_id.insert(0, sp.PieceToId('<s>'))
        if eos:
            sentence_id.append(sp.PieceToId('</s>'))
        return np.array(sentence_id, dtype=np.int32)

    def id2word(self, sentence_id):
        return self.vocab.DecodeIds(sentence_id.tolist())


class Iterator:
    def __init__(self, src_text, src_label, trg_text, src_vocab, trg_vocab, batch_size, gpu_id, sort=True, shuffle=True):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.size = len(src_text)

        self.sort = sort
        self.shuffle = shuffle

        self.batches = self._prepare_minibatch(src_text, src_label, trg_text, batch_size, gpu_id)

    def _convert(self, src, trg, label):
        src_id = [self.src_vocab.word2id(s) for s in src]
        trg_sos = self.trg_vocab.word2id(trg, sos=True)
        trg_eos = self.trg_vocab.word2id(trg, eos=True)
        return src_id, trg_sos, trg_eos, label

    def _prepare_minibatch(self, src, label, trg, batch_size, gpu_id):
        data = []
        for s, l, t in zip(src, label, trg):
            data.append(self._convert(s, t, l))

        if self.sort:
            data = sorted(data, key=lambda x: len(x[0]), reverse=True)
        batches = [convert.convert(data[b * batch_size: (b + 1) * batch_size], gpu_id) for b in range(len(data) // batch_size)]
        if len(data) % batch_size != 0:
            batches.append(convert.convert(data[-(len(data) % batch_size):], gpu_id))

        return batches

    def generate(self):
        batches = self.batches
        if self.shuffle:
            batches = random.sample(batches, len(batches))

        for batch in batches:
            yield batch


class MixIterator:
    def __init__(self, iterator1, iterator2, shuffle=True, multiple=1):
        # iterator1を大きいデータサイズのiteratorに指定する
        weight = 1 / (iterator1.size // iterator2.size)

        self.batches = []
        for batch in iterator1.batches:
            self.batches.append([batch, weight])
        for i in range(multiple):
            for batch in iterator2.batches:
                self.batches.append([batch, 1.0])

        self.shuffle = shuffle

    def generate(self):
        batches = self.batches
        if self.shuffle:
            batches = random.sample(batches, len(batches))

        for batch in batches:
            yield batch[0], batch[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()

    model_dir = args.model_dir
    if 'normal' in model_dir:
        vocab_type = 'normal'
    else:
        vocab_type = 'subword'
    """LOAD CONFIG FILE"""
    config_files = glob.glob(os.path.join(model_dir, '*.ini'))
    assert len(config_files) == 1, 'Put only one config file in the directory'
    config_file = config_files[0]
    config = configparser.ConfigParser()
    config.read(config_file)

    gpu_id = -1
    data_type = model_dir.split('_')[2]
    reg = False if data_type == 'l' or data_type == 's' else True

    vocab_size = int(config['Parameter']['vocab_size'])

    if data_type == 'l':
        section = 'Local'
    elif data_type == 'lr':
        section = 'Local_Reg'
    elif data_type == 's':
        section = 'Server'
    else:
        section = 'Server_Reg'
    train_src_file = config[section]['train_src_file']
    train_trg_file = config[section]['train_trg_file']
    valid_src_file = config[section]['valid_src_file']
    valid_trg_file = config[section]['valid_trg_file']
    test_src_file = config[section]['test_src_file']

    if vocab_type == 'normal':
        src_vocab = VocabNormal(reg)
        trg_vocab = VocabNormal(reg)
        if os.path.isfile(model_dir + 'src_vocab.normal.pkl') and os.path.isfile(model_dir + 'trg_vocab.normal.pkl'):
            src_vocab.load(model_dir + 'src_vocab.normal.pkl')
            trg_vocab.load(model_dir + 'trg_vocab.normal.pkl')
        else:
            init_vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
            src_vocab.build(train_src_file, True,  init_vocab, vocab_size)
            trg_vocab.build(train_trg_file, False, init_vocab, vocab_size)
            save_pickle(model_dir + 'src_vocab.normal.pkl', src_vocab.vocab)
            save_pickle(model_dir + 'trg_vocab.normal.pkl', trg_vocab.vocab)
        src_vocab.set_reverse_vocab()
        trg_vocab.set_reverse_vocab()

        sos = convert.convert_list(np.array([src_vocab.vocab['<s>']], dtype=np.int32), gpu_id)
        eos = convert.convert_list(np.array([src_vocab.vocab['</s>']], dtype=np.int32), gpu_id)

    elif vocab_type == 'subword':
        src_vocab = VocabSubword()
        trg_vocab = VocabSubword()
        if os.path.isfile(model_dir + 'src_vocab.sub.model') and os.path.isfile(model_dir + 'trg_vocab.sub.model'):
            src_vocab.load(model_dir + 'src_vocab.sub.model')
            trg_vocab.load(model_dir + 'trg_vocab.sub.model')
        else:
            src_vocab.build(train_src_file + '.sub', model_dir + 'src_vocab.sub', vocab_size)
            trg_vocab.build(train_trg_file + '.sub', model_dir + 'trg_vocab.sub', vocab_size)

        sos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('<s>')], dtype=np.int32), gpu_id)
        eos = convert.convert_list(np.array([src_vocab.vocab.PieceToId('</s>')], dtype=np.int32), gpu_id)

    src_file = test_src_file
    trg_file = train_trg_file

    if reg:
        label, src = load_with_label_reg(src_file)
    else:
        label, src = load_with_label(src_file)
    trg = load(trg_file)

    for i, (s, t) in enumerate(zip(src, trg), start=1):
        print('{} src'.format(i))
        for ss in s:
            print(ss)
            print(src_vocab.word2id(ss, sos, eos))
        # print('{} trg'.format(i))
        # print(t)
        # print(trg_vocab.word2id(t))