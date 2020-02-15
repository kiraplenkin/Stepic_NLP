import ast
import sys
import numpy as np


def update_ft_weights(center_embeddinds, context_embeddings, center_subword, context_word, label, learning_rate):
    w = center_embeddinds[center_subword].mean(0).copy()
    x = context_embeddings[context_word].copy()
    sigmoid = 1 / (1 + np.exp(-np.sum(w*x)))

    center_embeddinds[center_subword] = center_embeddinds[center_subword] - (learning_rate * x * (sigmoid - label)) / center_subword.shape[0]
    context_embeddings[context_word] = x - learning_rate * w * (sigmoid - label)
