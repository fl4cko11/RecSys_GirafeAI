"""EASE closed-form ridge reconstruction, no pretrained algorithms."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class EASE:
    def __init__(self, regularization=300):
        self.regularization = regularization

    def fit(self, ratings):
        self.x = (ratings >= 4).astype(np.float64).tocsr()
        gram = (self.x.T @ self.x).toarray()
        gram.flat[::gram.shape[0]+1] += self.regularization
        inverse = cho_solve(cho_factor(gram), np.eye(gram.shape[0]), check_finite=False)
        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.x[users] @ self.weights
