"""Numbers for the "Value network and correlated positions" example.

A toy version of the overfitting problem in "Reinforcement learning of value
networks". Each game has a hidden feature vector g. Every position of the game
is g plus a little noise, and every position carries the game's single outcome
z in {-1, +1}. We fit the same least-squares model on two training sets of the
same size and test on positions from fresh games.

Run: python3 scripts/value_overfit.py
"""

import numpy as np

D = 100          # feature dimension
SIGMA = 0.3      # per-position noise around the game's feature vector
N_TRAIN = 400    # positions in each training set
RIDGE = 1e-3


def make_games(rng, n_games, positions_per_game, w):
    g = rng.normal(size=(n_games, D))
    p_win = 1.0 / (1.0 + np.exp(-(g @ w)))
    z = np.where(rng.random(n_games) < p_win, 1.0, -1.0)
    x = g[:, None, :] + SIGMA * rng.normal(size=(n_games, positions_per_game, D))
    X = x.reshape(-1, D)
    Z = np.repeat(z, positions_per_game)
    return X, Z


def fit(X, Z):
    return np.linalg.solve(X.T @ X + RIDGE * np.eye(D), X.T @ Z)


def mse(theta, X, Z):
    return float(np.mean((X @ theta - Z) ** 2))


rows = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=D) / np.sqrt(D) * 2.0
    Xte, Zte = make_games(rng, 500, 1, w)
    out = []
    for per_game in (20, 1):
        X, Z = make_games(rng, N_TRAIN // per_game, per_game, w)
        th = fit(X, Z)
        out += [mse(th, X, Z), mse(th, Xte, Zte)]
    rows.append(out)
m = np.mean(rows, axis=0)
print("training set                    train MSE   test MSE")
print(f"20 positions from each of 20 games   {m[0]:.3f}      {m[1]:.3f}")
print(f"1 position from each of 400 games    {m[2]:.3f}      {m[3]:.3f}")
print("(mean over 20 seeds; a predictor that always outputs 0 has MSE 1.000)")
