import numpy as np

V, H = 4, 24  # vocabulary size, state size


def make_encoder(rng, H=H, V=V):
    """Finished: a random orthogonal A and a random embedding E."""
    A, _ = np.linalg.qr(rng.normal(size=(H, H)))
    E = rng.normal(size=(H, V)) / np.sqrt(V)
    return A, E


def encode(tokens, E, A):
    """v = h_T for h_t = A h_{t-1} + E[:, x_t], h_0 = 0."""
    h = np.zeros(E.shape[0])
    for t in tokens:
        h = A @ h + E[:, t]
    return h


def fit_readout(Hs, Y, ridge=1e-6):
    """Ridge regression: the (H, k) matrix W minimizing |Hs W - Y|^2 + ridge |W|^2."""
    d = Hs.shape[1]
    return np.linalg.solve(Hs.T @ Hs + ridge * np.eye(d), Hs.T @ Y)


def token_accuracy(W, Hs, tokens, V=V):
    """Fraction of source positions whose arg-max readout equals the true token."""
    n = tokens.shape[1]
    pred = (Hs @ W).reshape(len(Hs), n, V).argmax(-1)
    return float((pred == tokens).mean())


def experiment(n, seed=0, n_train=1500, n_test=500):
    """Finished: held-out token accuracy at source length n."""
    rng = np.random.default_rng(seed)
    A, E = make_encoder(rng)
    tr = rng.integers(0, V, size=(n_train, n))
    te = rng.integers(0, V, size=(n_test, n))
    Htr = np.stack([encode(s, E, A) for s in tr])
    Hte = np.stack([encode(s, E, A) for s in te])
    Y = np.eye(V)[tr].reshape(n_train, n * V)
    W = fit_readout(Htr, Y)
    return token_accuracy(W, Hte, te)
