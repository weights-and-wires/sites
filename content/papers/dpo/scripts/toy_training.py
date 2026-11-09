"""Train DPO on a three-response toy problem and compare with the closed form.

The preference data are exact Bradley-Terry counts for the true reward r* = [0, 1, 2],
so the optimum is known: pi*(y) proportional to pi_ref(y) exp(r*(y) / beta) (Equation 4).

Run: python3 content/papers/dpo/scripts/toy_training.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

ref = np.array([0.70, 0.25, 0.05])
r_star = np.array([0.0, 1.0, 2.0])


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bt_counts(r, n=1000.0):
    """C[i, j] = how often response i beat response j, in expectation."""
    C = n * sigmoid(r[:, None] - r[None, :])
    np.fill_diagonal(C, 0.0)
    return C


def mean_grad(theta, ref_logits, C, beta):
    lp, lr = np.log(softmax(theta)), np.log(softmax(ref_logits))
    rhat = beta * (lp - lr)
    g = np.zeros(3)
    for i in range(3):
        for j in range(3):
            if C[i, j] == 0:
                continue
            weight = sigmoid(rhat[j] - rhat[i])
            e = np.zeros(3)
            e[i], e[j] = 1.0, -1.0
            g += C[i, j] * (-beta * weight * e)
    return g / C.sum()


def mean_loss(theta, ref_logits, C, beta):
    lp, lr = np.log(softmax(theta)), np.log(softmax(ref_logits))
    rhat = beta * (lp - lr)
    total = 0.0
    for i in range(3):
        for j in range(3):
            if C[i, j]:
                total += C[i, j] * np.logaddexp(0.0, -(rhat[i] - rhat[j]))
    return total / C.sum()


def train(ref_logits, C, beta, lr, steps):
    theta = ref_logits.copy()
    for _ in range(steps):
        theta = theta - lr * mean_grad(theta, ref_logits, C, beta)
    return theta


C = bt_counts(r_star)
print("counts C[i, j] (row beat column):")
print(C)
ref_logits = np.log(ref)

for beta, steps in ((1.0, 3000), (0.5, 3000)):
    lr = 8.0 / beta**2
    theta = train(ref_logits, C, beta, lr, steps)
    pi = softmax(theta)
    w = ref * np.exp(r_star / beta)
    closed = w / w.sum()
    rhat = beta * np.log(pi / ref)
    print()
    print(f"beta = {beta}, lr = {lr}, steps = {steps}")
    print("  initial loss        =", round(float(mean_loss(ref_logits, ref_logits, C, beta)), 4))
    print("  final loss          =", round(float(mean_loss(theta, ref_logits, C, beta)), 4))
    print("  trained pi          =", pi.tolist())
    print("  closed-form pi*     =", closed.tolist())
    print("  max |pi - pi*|      =", float(np.abs(pi - closed).max()))
    print("  implicit reward     =", rhat.tolist())
    print("  differences rhat_1 - rhat_0, rhat_2 - rhat_1 =", float(rhat[1] - rhat[0]), float(rhat[2] - rhat[1]), "(true: 1, 1)")
    print("  ranking by pi/ref   =", np.argsort(-(pi / ref)).tolist(), "(true: [2, 1, 0])")
