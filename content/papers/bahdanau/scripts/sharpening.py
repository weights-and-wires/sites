"""Train an additive alignment model on a reversal task and watch the weights sharpen.

Each decoder step i has a one-hot "position" state s_{i-1}. Each source position j has a
one-hot annotation h_j. The target for step i is the context h_{T-1-i}, so the right
alignment is the anti-diagonal. Only the alignment model (W_a, U_a, v_a) is trained.
The gradient uses the softmax backward rule from exercise 4. The initial weights have a
standard deviation of 0.3, so the alignment starts near uniform. With a standard
deviation of 0.01 the product of tiny weights makes the gradient too small to move in 300 steps.

Run: python3 scripts/sharpening.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

T, n_prime = 5, 16
rng = np.random.default_rng(0)
S = np.eye(T)  # row i is s_{i-1}
H = np.eye(T)  # row j is h_j
Wa = rng.normal(scale=0.3, size=(n_prime, T))
Ua = rng.normal(scale=0.3, size=(n_prime, T))
va = rng.normal(scale=0.3, size=n_prime)
target = np.arange(T)[::-1]  # step i should attend to source T-1-i


def forward():
    U = np.tanh((S @ Wa.T)[:, None, :] + (H @ Ua.T)[None, :, :])  # (T_y, T_x, n')
    e = U @ va
    e = e - e.max(axis=1, keepdims=True)
    alpha = np.exp(e) / np.exp(e).sum(axis=1, keepdims=True)
    return U, alpha


def entropy(alpha):
    return float(-(alpha * np.log(alpha)).sum(axis=1).mean())


lr = 0.5
for step in range(0, 301):
    U, alpha = forward()
    loss = -np.log(alpha[np.arange(T), target]).mean()
    if step in (0, 50, 100, 200, 300):
        print(f"step {step:3d}  loss {loss:.4f}  entropy {entropy(alpha):.4f}  "
              f"mean max weight {alpha.max(axis=1).mean():.4f}  "
              f"row sums {alpha.sum(axis=1).min():.4f}..{alpha.sum(axis=1).max():.4f}")
    # Cross-entropy on alpha: dL/de_ij = (alpha_ij - onehot_ij) / T_y.
    onehot = np.zeros_like(alpha)
    onehot[np.arange(T), target] = 1.0
    de = (alpha - onehot) / T
    dva = np.einsum("ij,ijk->k", de, U)
    dpre = de[:, :, None] * va[None, None, :] * (1 - U**2)  # (T_y, T_x, n')
    dWa = np.einsum("ijk,ia->ka", dpre, S)
    dUa = np.einsum("ijk,ja->ka", dpre, H)
    va -= lr * dva
    Wa -= lr * dWa
    Ua -= lr * dUa

print("\nfinal alpha (rows: target step i, columns: source position j):")
print(forward()[1])
print("argmax per row:", forward()[1].argmax(axis=1), " want:", target)
