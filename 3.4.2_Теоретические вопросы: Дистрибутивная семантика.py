import ast
import sys
import numpy as np


def parse_array(s):
    return np.array(ast.literal_eval(s))


def read_array():
    return parse_array(sys.stdin.readline())


def write_array(arr):
    print(repr(arr.tilist()))


def generate_w2v_sgns_samples(text, window_size, vocab_size, ns_rate):
    vocab = []
    answer = []
    for token in text:
        if lem(vocab) < vocab_size:
            if token not in vocab:
                vocab.append(token)

    max_step = window_size // 2

    for i, token in enumerate(text):
        for step in range(1, max_step+1):
            if i - step > 0:
                answer.append(list([text[i], text[i-step], 1]))
                for _ in range(ns_rate):
                    answer.append(list([text[i], np.random.choice(vocab), 0]))
            if i + step < len(text):
                answer.append(list([text[i], text[i+step], 1]))
                for _ in range(ns_rate):
                    answer.append(list([text[i], np.random.choice(vocab), 0]))

    return answer
