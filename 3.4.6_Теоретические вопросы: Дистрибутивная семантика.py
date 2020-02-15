import ast
import sys
import numpy as np
from scipy.sparse import dok_matrix


def generate_coocurence_matrix(texts, vocab_size):
    result = dok_matrix((vocab_size, vocab_size), dtype=np.int8)
    for vokab_token in range(vocab_size):
        for text in texts:
            if vocab_token in text:
                for token in set(text):
                    if vocab_token != token:
                        result[vocab_token, token] += 1

    return result
