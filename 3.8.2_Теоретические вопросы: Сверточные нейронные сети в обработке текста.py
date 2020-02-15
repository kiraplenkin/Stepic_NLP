import ast
import sys
import numpy as np
from scipy.sparse import dok_matrix


def apply_convolution(data, kernel, bias):
    answer = []
    for window in range(data.shape[0] - kernel.shape[2] + 1):
        matrix = dok_matrix((data.shape[1], kernel.shape[0]), np.float)
        for i, data_line in enumerate(data.T):
            for j, kernnel_line in enumerate(kernel):
                matrix[i, j] = np.dot(data_line[window:kernel.shape[2]+window], kernnel_line[i])
        answer = np.append(answer, np.sum(matrix.toarray(), axis=0) + bias)
    
    answer = answer.reshape(int(answer.shape[0] / kernel.shape[0]), kernel.shape[0])
    return answer
