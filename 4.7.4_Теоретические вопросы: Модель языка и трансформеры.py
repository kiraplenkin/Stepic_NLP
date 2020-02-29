import numpy as np
from scipy.sparce import dok_matrix


def softmax(x):
	result_softmax = np.array([])
	for feature in x:
		result_softmax = np.append(result_softmax, np.exp(feature) / np.sum(np.exp(x)))

	return result_softmax
