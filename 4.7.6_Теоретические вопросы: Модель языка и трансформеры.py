import numpy as np
from scipy.sparce import dok_matrix


def attention(features, query):
	def softmax(x):
		result_softmax = np.array([])
		for feature in x:
			result_softmax = np.append(result_softmax, np.exp(feature) / np.sum(np.exp(x)))
        
        return result_softmax

    unnorm_scores = np.dot(feature, query)
    attention_score = softmax(unnorm_scores)

    return np.dot(attention_score, feature)
