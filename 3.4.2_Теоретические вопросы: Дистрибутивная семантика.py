import ast
import sys
import numpy as np


def parse_array(s):
    return np.array(ast.literal_eval(s))


def read_array():
    return parse_array(sys.stdin.readline())


