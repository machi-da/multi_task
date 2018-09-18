import sys
import numpy as np
import logging
from logging import getLogger
from gensim.models import KeyedVectors, word2vec
from tqdm import tqdm

"""
東北大学乾研の学習済みword2vecモデルを使用
http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
"""


class Word2Vec:
    def __init__(self):
        self.logger = getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

    def train(self, text_file, size=256, window=5, sample=0.001, negative=5, hs=0):
        logger = self.logger
        logger.info('Load text_file')
        data = word2vec.LineSentence(text_file)
        logger.info('Start training')
        model = word2vec.Word2Vec(data, size=size, window=window, sample=sample, negative=negative, hs=hs)
        logger.info('Finish training')
        model_name = '{}.w2v'.format(text_file)
        model.wv.save_word2vec_format(model_name + '.txt')
        model.wv.save_word2vec_format(model_name + '.bin', binary=True)
        logger.info('Save model: {} and {}'.format(model_name + '.txt', model_name + '.bin'))

    def make_initialW(self, vocab, model_file='/home/lr/machida/entity_vector/entity_vector.model.bin'):
        """
        model_file: wikipedia学習済みモデル
        """

        vocab_size = len(vocab)
        embeddings_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
        vector_size = embeddings_model.vector_size

        # 正規分布で初期化
        embedding_matrix = np.random.normal(0.0, 1.0, (vocab_size, vector_size))
        # ゼロベクトルで初期化
        # embedding_matrix = np.zeros((vocab_size, vector_size))

        for word, index in tqdm(vocab.items()):
            word_markup = '[' + word + ']'
            if word in embeddings_model.index2word:
                embedding_matrix[index] = embeddings_model[word]
            elif word_markup in embeddings_model.index2word:
                embedding_matrix[index] = embeddings_model[word_markup]

        return embedding_matrix, vector_size


def check(w2v):
    vocab = {
        'ハワイロア': 0
    }
    embedding_matrix, _ = w2v.make_initialW(vocab)
    print(embedding_matrix)


def train(w2v, text_file):
    w2v.train(text_file)


def most_similar(model, word):
    embeddings_model = KeyedVectors.load_word2vec_format(model, binary=True)
    print(embeddings_model.most_similar(word))


if __name__ == '__main__':
    args = sys.argv
    w2v = Word2Vec()

    if len(args) == 2:
        train(w2v, args[1])
    elif len(args) == 3:
        most_similar(args[1], args[2])
