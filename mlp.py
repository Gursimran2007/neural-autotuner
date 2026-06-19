"""
The MOAT: our own from-scratch neural net that predicts the winning kernel
config from the problem shape -- in one forward pass, no search.

A small MLP classifier over kernel.TILES, with manual forward/backprop in pure
NumPy (no PyTorch). Trained on the self-generated benchmark data. At inference it
turns the expensive brute-force search (benchmark every config) into a single
cheap prediction, which the oracle then VERIFIES.
"""
import numpy as np


def _he(shape, rng):
    return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / shape[0])


class MLP:
    def __init__(self, n_in, n_out, hidden=(64, 64), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [_he((sizes[i], sizes[i + 1]), rng) for i in range(len(sizes) - 1)]
        self.b = [np.zeros((1, sizes[i + 1]), np.float32) for i in range(len(sizes) - 1)]

    def forward(self, X, cache=False):
        a = X
        acts, pres = [a], []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            pres.append(z)
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z   # ReLU; last linear
            acts.append(a)
        if cache:
            self._acts, self._pres = acts, pres
        return a  # logits

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def loss_and_grad(self, X, y, sample_w=None, l2=1e-4):
        """Cross-entropy (optionally per-sample weighted) + L2. Returns (loss, grads)."""
        n = X.shape[0]
        logits = self.forward(X, cache=True)
        P = self.softmax(logits)
        w = np.ones(n, np.float32) if sample_w is None else sample_w
        w = w / w.mean()
        ce = -np.log(P[np.arange(n), y] + 1e-9)
        loss = float((w * ce).mean())
        # backprop
        dZ = P.copy()
        dZ[np.arange(n), y] -= 1
        dZ *= (w[:, None] / n)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            a_prev = self._acts[i]
            gW[i] = a_prev.T @ dZ + l2 * self.W[i]
            gb[i] = dZ.sum(axis=0, keepdims=True)
            if i > 0:
                dA = dZ @ self.W[i].T
                dZ = dA * (self._pres[i - 1] > 0)
        return loss, (gW, gb)

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def step(self, grads, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam update. m, v are running moment lists matching params."""
        gW, gb = grads
        params = self.W + self.b
        gs = gW + gb
        for i, (p, g) in enumerate(zip(params, gs)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
            mh = m[i] / (1 - beta1 ** t)
            vh = v[i] / (1 - beta2 ** t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def zeros_like_params(self):
        return [np.zeros_like(p) for p in (self.W + self.b)]
