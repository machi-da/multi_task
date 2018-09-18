import sys
import numpy as np
from gensim.models import KeyedVectors, word2vec
from tqdm import tqdm

"""
東北大学乾研の学習済みword2vecモデルを使用
http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
"""


class Word2Vec:
    def train(self, text_file, size=256, window=5, sample=0.001, negative=5, hs=0):
        print('Load text file')
        data = word2vec.LineSentence(text_file)
        print('Start training')
        model = word2vec.Word2Vec(data, size=size, window=window, sample=sample, negative=negative, hs=hs)
        print('Finish training')
        model_name = '{}.w2v'.format(text_file)
        model.wv.save_word2vec_format(model_name)
        print('Save model: {}'.format(model_name))

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


if __name__ == '__main__':
    w2v = Word2Vec()
    # vocab = {
    #     'ハワイロア': 0
    # }
    # embedding_matrix, _ = w2v.make_initialW(vocab)
    # print(embedding_matrix)

    args = sys.argv
    w2v.train(args[1])