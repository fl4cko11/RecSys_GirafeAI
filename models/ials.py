"""Weighted regularized matrix factorization with explicit ALS updates."""
import numpy as np
from scipy.linalg import solve

class IALS:
    def __init__(self, factors=32, regularization=10, alpha=10, iterations=10, seed=42):
        self.factors, self.regularization = factors, regularization
        self.alpha, self.iterations, self.seed = alpha, iterations, seed

    def targets(self, values):
        return (values >= 4).astype(np.float64)

    def _update(self, ratings, fixed):
        gram = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        result = np.zeros((ratings.shape[0], self.factors))
        for row in range(ratings.shape[0]):
            a, b = ratings.indptr[row:row+2]
            ids, values = ratings.indices[a:b], ratings.data[a:b]
            if len(ids):
                vectors = fixed[ids]
                lhs = gram + self.alpha * (vectors.T @ vectors)
                rhs = vectors.T @ ((1 + self.alpha) * self.targets(values))
                result[row] = solve(lhs, rhs, assume_a='pos', check_finite=False)
        return result

    def fit(self, ratings):
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, .05, (ratings.shape[1], self.factors))
        transposed = ratings.T.tocsr()
        for _ in range(self.iterations):
            self.users = self._update(ratings, self.items)
            self.items = self._update(transposed, self.users)
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
