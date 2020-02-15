import ast
import sys
import numpy as np


def generate_ft_sgns_samples(text, window_size, vocab_size, ns_rate, token2subwords):
    vocab = []
    answer = []

    for token in text:
        if len(vocab) < vocab_size:
            if token not in vocab:
                vocab.append(token)

    max_step = window_size // 2

    for i, token in enumerate(text):
        for step in range(1, max_step+1):
            if i - step >= 0:
                ngr = []
                ngr.append(token)
                ngr.append(set(token2subwords[token]))
                for n in range(len(token2subwords[token])):
                    if token2subwords[token][n] not in ngr:
                        ngr.append(token2subwords[token][n])
                answer.append(list([list(ngr), text(i-step), 1]))
                for _ in range(ns_rate):
                    answer.append(list([list(ngr), np.random.choice(vocab), 0]))
            if i + step < len(text):
                ngr = []
                ngr.append(token)
                ngr.append(set(token2subwords[token]))
                for n in range(len(token2subwords[token])):
                    if token2subwords[token][n] not in ngr:
                        ngr.append(token2subwords[token][n])
                answer.append(list([list(ngr), text(i+step), 1]))
                for _ in range(ns_rate):
                    answer.append(list([list(ngr), np.random.choice(vocab), 0]))
    return answer
