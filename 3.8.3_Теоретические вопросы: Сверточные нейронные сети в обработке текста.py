import ast
import sys
import numpy as np
from scipy.sparse import dok_matrix


def calculate_kernel_grad(x, y, kernel, bias):
    inLen, _ = x.shape
    outChannels, inChannels, kernel_size = kernel.shape
    outLen, outChannels, = y.shape

    grad = np.zeros((outChannels, inChannels, kernel_size))

    for pos in np.arange(outLen):
        for oc in np.arange(outChannels):
            for k in np.arange(kernel_size):
                for ic in np.arange(inChannels):
                    grad[oc, ic, k] += x[pos + k, ic]
    return grad
