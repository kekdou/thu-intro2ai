import torch
import numpy as np

from gensim.models import KeyedVectors

def build_embedding_weight(word_to_id, file_path):
    w2v_model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    vocab_size = len(word_to_id)
    embedding_dim = 50
    weight_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    for word, idx in word_to_id.items():
        if word in w2v_model:
            weight_matrix[idx] = w2v_model[word]
    weight_matrix[0] = np.zeros(embedding_dim)
    return torch.tensor(weight_matrix, dtype=torch.float32)
    