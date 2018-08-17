import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

"""
東北大学乾研の学習済みword2vecモデルを使用
http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
"""


def make_initialW(vocab, hidden_size=200):
    vocab_size = len(vocab)
    embeddings_model = KeyedVectors.load_word2vec_format('/home/lr/machida/entity_vector/entity_vector.model.bin', binary=True)

    # 正規分布で初期化
    embedding_matrix = np.random.normal(0.0, 1.0, (vocab_size, hidden_size))
    # ゼロベクトルで初期化
    # embedding_matrix = np.zeros((vocab_size, hidden_size))

    for word, index in tqdm(vocab.items()):
        word_markup = '[' + word + ']'
        if word in embeddings_model.index2word:
            embedding_matrix[index] = embeddings_model[word]
        elif word_markup in embeddings_model.index2word:
            embedding_matrix[index] = embeddings_model[word_markup]

    return embedding_matrix


if __name__ == '__main__':
    vocab = {
        'ハワイロア': 0
    }
    embedding_matrix = make_initialW(vocab)
    print(embedding_matrix)