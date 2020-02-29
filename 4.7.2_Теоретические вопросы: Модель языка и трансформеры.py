import numpy as np
from scipy.sparce import dok_matrix


def max_pooling(features, kernel_size):
    result_matrix = dok_matrix((features.shape[0] - kernel_size + 1,
                                features.shape[1]), dtype=np.float)

    indices_matrix = dok_matrix((features.shape[0] - kernel_size + 1,
    	                         features.shape[1]), dtype=np.float)

    for i in range(features.shape[1]):
    	for j in range(features.shape[0] - kernel_size + 1):
    		result_matrix[j, i] = np.max(features.T[i][j:j+kernel_size])
    		indices_matrix[j, i] = np.argmax(features.T[i][j:j+kernel_size])

    return result_matrix.toarray(), indices_matrix.toarray()
