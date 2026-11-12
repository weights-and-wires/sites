"""Numbers for the "Exploration noise at the root" example.

P = (1 - eps) * p + eps * eta, with eta ~ Dir(alpha) and eps = 0.25. The paper
sets alpha = 0.3 for chess, 0.15 for shogi, and 0.03 for Go (Methods,
"Configuration"). The legal-move counts below are rough figures: 35 for chess
and 250 for Go come from the AlphaGo paper's introduction, and 80 for shogi is
an approximation that is not in the AlphaZero paper.

Run: python3 scripts/root_noise.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
EPS = 0.25


def add_root_noise(p, eps, alpha, rng):
    p = np.asarray(p, dtype=float)
    return (1 - eps) * p + eps * rng.dirichlet(np.full(len(p), alpha))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = np.array([0.90, 0.06, 0.03, 0.01])
    print(f"network prior p            = {p}")
    for alpha in (0.3, 3.0):
        mixed = add_root_noise(p, EPS, alpha, rng)
        print(f"one draw, alpha = {alpha:<4}   = {mixed}  (sum {mixed.sum():.4f})")
    draws = np.array([add_root_noise(p, EPS, 0.3, rng) for _ in range(20000)])
    print(f"mean of 20000 draws        = {draws.mean(axis=0)}")
    print(f"(1 - eps) p + eps / K      = {(1 - EPS) * p + EPS / len(p)}")
    print(f"floor on any prior (1-eps)p: move with p = 0.01 keeps at least {(1 - EPS) * 0.01:.4f}")

    print("\nscale alpha with the number of legal moves (alpha * K stays between 7 and 12):")
    for game, alpha, K in (("chess", 0.3, 35), ("shogi", 0.15, 80), ("Go", 0.03, 250)):
        eta = rng.dirichlet(np.full(K, alpha), size=4000)
        top = np.sort(eta, axis=1)[:, -1].mean()
        print(f"  {game:5s}: alpha = {alpha:<4}, K = {K:3d}, alpha*K = {alpha * K:5.2f}, "
              f"largest noise share = {top:.3f}")
