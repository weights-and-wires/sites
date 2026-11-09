"""Worked example for the DeepSeekMoE gate (Equations 9-11) and balance losses (Equations 12-17).

Run: python3 scripts/gate_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# One token, 6 routed experts, K' = 3 active. Router logits u^T e_i.
logits = np.array([2.0, 0.5, 1.0, -1.0, 0.0, 1.5])
s = softmax(logits)
top = np.argsort(-logits)[:3]
print("s_i,t (Eq. 11)        =", s)
print("top-3 routed experts  =", top.tolist())
print("gates g_i,t (Eq. 10)  =", np.where(np.isin(np.arange(6), top), s, 0.0))
print("sum of gates          =", round(float(s[top].sum()), 4))
print("renormalized (Mixtral-style) =", s[top] / s[top].sum())

# Layer output with scalar stand-in experts. u = 1.0, shared FFN returns 100, routed FFN_i returns 10*(i+1).
shared_out = 100.0
routed_out = 10.0 * (np.arange(6) + 1)
gates = np.where(np.isin(np.arange(6), top), s, 0.0)
h = 1.0 + shared_out + float((gates * routed_out).sum())
h_renorm = 1.0 + shared_out + float((s[top] / s[top].sum() * routed_out[top]).sum())
print("h_t, paper gates      =", round(h, 4))
print("h_t, renormalized     =", round(h_renorm, 4))
print()

# Balance losses: T = 4 tokens, N' = 4 routed experts, K' = 2, alpha = 1.
alpha = 1.0
n, k = 4, 2


def losses(s, idx, groups):
    t = s.shape[0]
    f = np.array([(idx == i).any(axis=1).sum() for i in range(n)]) * n / (k * t)
    p = s.mean(axis=0)
    exp_loss = alpha * float((f * p).sum())
    f_dev = np.array([f[g].mean() for g in groups])
    p_dev = np.array([p[g].sum() for g in groups])
    return f, p, exp_loss, f_dev, p_dev, alpha * float((f_dev * p_dev).sum())


groups = [[0, 1], [2, 3]]
bal_s = np.full((4, 4), 0.25)
bal_idx = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])
col_s = np.tile([0.55, 0.30, 0.10, 0.05], (4, 1))
col_idx = np.tile([0, 1], (4, 1))
cases = [
    ("balanced", bal_s, bal_idx, groups),
    ("collapsed, experts 0-1 on device 1", col_s, col_idx, groups),
    ("collapsed, experts 0,2 on device 1", col_s, col_idx, [[0, 2], [1, 3]]),
]
for name, s_, i_, g_ in cases:
    print(name, "with groups", g_)
    f, p, le, fd, pd, ld = losses(s_, i_, g_)
    print(f"          f = {f}, P = {p}")
    print(f"          L_ExpBal = {le:.4f}")
    print(f"          f' = {fd}, P' = {pd}, L_DevBal = {ld:.4f}")
