import ast
import sys
import numpy as np


def update_w2v_weights(center_embeddings, context_embeddings, center_word, context_word, label, learning_rate):
    w = center_embeddings[center_word].copy()
    x = context_embeddings[context_word].copy()

    sigmoid = 1 / (1 + np.exp(-np.sum(w*x)))
    center_embeddings[center_word] = w - learning_rate * x * (sigmoid - label)
    context_word[context_word] = x - learning_rate * w * (sigmoid - label)