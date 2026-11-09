"""Auxiliary-loss-free load balancing on a toy router (Section 2.1.2, Equations 12-16).

The affinities are made up: every expert has a fixed preference offset, so
without balancing some experts get far more than their share of tokens.
This is not the paper's data. gamma = 0.001 is the paper's bias update speed
(Section 4.2); the batch, expert count, and step count are small on purpose.
"""
import numpy as np

N_R, K_R, TOKENS, STEPS, GAMMA = 64, 8, 2048, 400, 0.001


def route(s, b, k):
    """Top-k on s + b (Equation 16). Returns the chosen expert indices per token."""
    return np.argsort(-(s + b), axis=1)[:, :k]


def gates(s, idx):
    """Equation 13: normalize the ORIGINAL affinities s over the selected experts."""
    chosen = np.take_along_axis(s, idx, axis=1)
    return chosen / chosen.sum(axis=1, keepdims=True)


def loads(idx, n):
    return np.bincount(idx.ravel(), minlength=n)


def update(b, load, gamma):
    """Lower the bias of overloaded experts by gamma, raise the underloaded ones."""
    return b - gamma * np.sign(load - load.mean())


def stats(load):
    """Coefficient of variation of the load and max load over mean load."""
    return load.std() / load.mean(), load.max() / load.mean()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tiny_example():
    s = np.array([[0.9, 0.8, 0.3, 0.2]])
    b = np.array([-0.5, 0.0, 0.2, 0.0])
    print("tiny example: s =", s[0], "bias =", b, "K = 2")
    for name, bias in (("without bias", np.zeros(4)), ("with bias   ", b)):
        idx = np.sort(route(s, bias, 2), axis=1)
        print(f"  {name}: experts {idx[0].tolist()}, gates {np.round(gates(s, idx)[0], 4).tolist()}")
    print("  s + b =", np.round(s[0] + b, 2))


def main():
    tiny_example()
    rng = np.random.default_rng(0)
    pref = rng.normal(0, 0.8, N_R)  # fixed per-expert preference
    b = np.zeros(N_R)
    print(f"\nN_r = {N_R}, K_r = {K_R}, {TOKENS} tokens per step, gamma = {GAMMA}")
    print("fair share per expert =", TOKENS * K_R / N_R)
    print("step  CV(load)  max/mean   (with bias)   |  CV(load)  max/mean (no bias)")
    base_cv = None
    for step in range(STEPS + 1):
        s = sigmoid(rng.normal(0, 1, (TOKENS, N_R)) + pref)
        load = loads(route(s, b, K_R), N_R)
        base = loads(route(s, np.zeros(N_R), K_R), N_R)
        if step in (0, 25, 50, 100, 200, 300, 400):
            cv, mx = stats(load)
            bcv, bmx = stats(base)
            print(f"{step:4d}  {cv:8.4f}  {mx:8.3f}                 |  {bcv:8.4f}  {bmx:8.3f}")
        b = update(b, load, GAMMA)
    print("final bias range: min %.4f, max %.4f" % (b.min(), b.max()))


if __name__ == "__main__":
    main()
