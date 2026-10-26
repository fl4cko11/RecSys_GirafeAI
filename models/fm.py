"""Second-order FM with categorical metadata and sampled BCE training."""
import numpy as np
import torch
from torch import nn

class FM:
    def __init__(self, item_features, n_features, factors=32, regularization=1e-5,
                 learning_rate=.003, epochs=8, negatives=2, metadata=True, seed=42):
        self.item_features = np.asarray(item_features, dtype=np.int64)
        self.n_features, self.factors = n_features, factors
        self.regularization, self.learning_rate = regularization, learning_rate
        self.epochs, self.negatives, self.metadata, self.seed = epochs, negatives, metadata, seed

    def fit(self, ratings):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_users, n_items = ratings.shape
        features = self.item_features if self.metadata else self.item_features[:, :1]
        # Feature 0 is padding, users 1..U, catalog features start at U+1.
        catalog = np.where(features > 0, features + self.n_users, 0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings = nn.Embedding(self.n_users+self.n_features+1, self.factors, padding_idx=0).to(device)
        linear = nn.Embedding(self.n_users+self.n_features+1, 1, padding_idx=0).to(device)
        nn.init.normal_(embeddings.weight, std=.03)
        nn.init.zeros_(linear.weight)
        with torch.no_grad(): embeddings.weight[0].zero_()
        bias = nn.Parameter(torch.zeros((), device=device))
        optimizer = torch.optim.Adam([*embeddings.parameters(), *linear.parameters(), bias], lr=self.learning_rate)
        co = ratings.tocoo()
        known = [set(ratings.indices[ratings.indptr[u]:ratings.indptr[u+1]]) for u in range(self.n_users)]
        user_ids, item_ids = co.row.astype(np.int64), co.col.astype(np.int64)
        labels = (co.data >= 4).astype(np.float32)
        catalog_t = torch.tensor(catalog, device=device)
        self.loss_history = []
        for _ in range(self.epochs):
            pos_users = np.repeat(user_ids[labels == 1], self.negatives)
            neg_items = rng.integers(n_items, size=len(pos_users))
            for j, u in enumerate(pos_users):
                while neg_items[j] in known[u]:
                    neg_items[j] = rng.integers(n_items)
            us = np.concatenate([user_ids, pos_users])
            it = np.concatenate([item_ids, neg_items])
            ys = np.concatenate([labels, np.zeros(len(pos_users), dtype=np.float32)])
            order = rng.permutation(len(us)); total = 0.
            for start in range(0, len(us), 4096):
                ix = order[start:start+4096]
                u = torch.as_tensor(us[ix]+1, device=device)
                i = torch.as_tensor(it[ix], device=device)
                ids = torch.cat([u[:, None], catalog_t[i]], dim=1)
                v = embeddings(ids)
                logits = bias + linear(ids).sum((1, 2)) + .5*((v.sum(1)**2-v.square().sum(1)).sum(1))
                target = torch.as_tensor(ys[ix], device=device)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
                loss = loss + self.regularization*(v.square().sum(2).mean()+linear(ids).square().mean())
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item()*len(ix)
            self.loss_history.append(total/len(us))
        with torch.no_grad():
            item_v = embeddings(catalog_t)
            self.items = item_v.sum(1).cpu().numpy()
            self.item_bias = (linear(catalog_t).sum((1,2)) + .5*(item_v.sum(1).square()-item_v.square().sum(1)).sum(1)).cpu().numpy()
            self.users = embeddings.weight[1:self.n_users+1].cpu().numpy().copy()
            self.user_bias = linear.weight[1:self.n_users+1, 0].cpu().numpy().copy()+bias.item()
        return self

    def score(self, users):
        return self.users[users] @ self.items.T + self.item_bias[None, :] + self.user_bias[users, None]
