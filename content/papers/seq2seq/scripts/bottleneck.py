"""A fixed-size state cannot hold arbitrarily long inputs.

This is a linear toy, not the paper's LSTM. The encoder is the linear recurrence
h_t = A h_{t-1} + E[:, x_t] with a random orthogonal A, and v = h_T is its final
state (Section 2). The decoder is a linear readout fitted by ridge regression that
recovers the source tokens from v. We measure per-token accuracy as the source
length n grows while the state size H stays fixed.
"""

import numpy as np

V, H = 4, 24  # vocabulary size, state size
N_TRAIN, N_TEST = 3000, 1000


def make_encoder(rng, H=H, V=V):
    A, _ = np.linalg.qr(rng.normal(size=(H, H)))
    E = rng.normal(size=(H, V)) / np.sqrt(V)
    return A, E


def encode(tokens, E, A):
    h = np.zeros(E.shape[0])
    for t in tokens:
        h = A @ h + E[:, t]
    return h


def fit_readout(Hs, Y, ridge=1e-6):
    d = Hs.shape[1]
    return np.linalg.solve(Hs.T @ Hs + ridge * np.eye(d), Hs.T @ Y)


def token_accuracy(W, Hs, tokens, V=V):
    n = tokens.shape[1]
    pred = (Hs @ W).reshape(len(Hs), n, V).argmax(-1)
    return float((pred == tokens).mean())


def experiment(n, seed=0):
    rng = np.random.default_rng(seed)
    A, E = make_encoder(rng)
    tr = rng.integers(0, V, size=(N_TRAIN, n))
    te = rng.integers(0, V, size=(N_TEST, n))
    Htr = np.stack([encode(s, E, A) for s in tr])
    Hte = np.stack([encode(s, E, A) for s in te])
    Y = np.eye(V)[tr].reshape(N_TRAIN, n * V)
    W = fit_readout(Htr, Y)
    return token_accuracy(W, Hte, te)


if __name__ == "__main__":
    print(f"state size H = {H}, vocabulary V = {V}, chance = {1 / V:.2f}")
    print(f"dimension count: n * (V - 1) = H at n = {H // (V - 1)}")
    for n in (2, 4, 6, 7, 8, 10, 12, 16, 24, 32):
        print(f"n={n:2d}  held-out token accuracy = {experiment(n):.3f}")
