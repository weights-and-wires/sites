import numpy as np


def _softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def loss_and_grads(Xi, Xt, Wi, Wt, t):
    """Return (loss, dWi, dWt, dt) for the symmetric contrastive loss.

    Xi: (n, d_i), Xt: (n, d_t), Wi: (d_i, d_e), Wt: (d_t, d_e), t: float.
    I_f = Xi @ Wi and T_f = Xt @ Wt are L2-normalized to I_e and T_e,
    logits = exp(t) * I_e @ T_e.T, and the loss is
    (cross entropy over rows + cross entropy over columns) / 2 with the
    diagonal as the target, each averaged over the n pairs.
    """
    n = len(Xi)
    scale = np.exp(t)
    If, Tf = Xi @ Wi, Xt @ Wt
    nI = np.linalg.norm(If, axis=1, keepdims=True)
    nT = np.linalg.norm(Tf, axis=1, keepdims=True)
    Ie, Te = If / nI, Tf / nT
    L = scale * (Ie @ Te.T)

    P_rows = _softmax(L, axis=1)
    P_cols = _softmax(L, axis=0)
    idx = np.arange(n)
    loss = (-np.log(P_rows[idx, idx]).mean() - np.log(P_cols[idx, idx]).mean()) / 2

    G = (P_rows + P_cols - 2 * np.eye(n)) / (2 * n)
    dIe = scale * (G @ Te)
    dTe = scale * (G.T @ Ie)
    dIf = (dIe - Ie * np.sum(Ie * dIe, axis=1, keepdims=True)) / nI
    dTf = (dTe - Te * np.sum(Te * dTe, axis=1, keepdims=True)) / nT
    dt = np.sum(G * L)
    return float(loss), Xi.T @ dIf, Xt.T @ dTf, float(dt)
