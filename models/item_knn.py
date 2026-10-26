"""Item cosine kNN, trained from positive implicit interactions."""
import numpy as np
from scipy import sparse

class ItemKNN:
    def __init__(self, neighbors=100, shrinkage=10):
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def fit(self, ratings):
        self.x = (ratings >= 4).astype(np.float64).tocsr()
        counts = np.asarray(self.x.sum(axis=0)).ravel()
        co = (self.x.T @ self.x).tocoo()
        sim = co.data / (np.sqrt(counts[co.row] * counts[co.col]) + self.shrinkage)
        sim[co.row == co.col] = 0
        matrix = sparse.csr_matrix((sim, (co.row, co.col)), shape=co.shape)
        # Keep incoming neighbors: a candidate is scored from its most similar items.
        matrix = matrix.tocsc()
        for col in range(matrix.shape[1]):
            a, b = matrix.indptr[col:col+2]
            if b-a > self.neighbors:
                indices = np.lexsort((matrix.indices[a:b], -matrix.data[a:b]))
                matrix.data[a:b][indices[self.neighbors:]] = 0
        matrix.eliminate_zeros()
        self.weights = matrix.tocsr()
        return self

    def score(self, users):
        return (self.x[users] @ self.weights).toarray()
