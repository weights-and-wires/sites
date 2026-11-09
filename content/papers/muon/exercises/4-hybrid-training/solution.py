import numpy as np


def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    """Finished for you: approximate U V^T for G = U S V^T."""
    a, b, c = coeffs
    X = G / (np.linalg.norm(G) + eps)
    tall = G.shape[0] > G.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.T
    return X


def make_problem(seed=0, n=256, d_in=16, d_h=32):
    """A toy regression task: a random 2-layer teacher produces the targets."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d_in))
    teacher_in = rng.normal(size=(d_in, d_h)) / np.sqrt(d_in)
    teacher_out = rng.normal(size=(d_h, 1)) / np.sqrt(d_h)
    Y = np.tanh(X @ teacher_in) @ teacher_out
    params = {
        "W_in": rng.normal(size=(d_in, d_h)) / np.sqrt(d_in),
        "W_h": rng.normal(size=(d_h, d_h)) / np.sqrt(d_h),
        "W_out": rng.normal(size=(d_h, 1)) / np.sqrt(d_h),
    }
    return X, Y, params


def loss_and_grads(params, X, Y):
    """Finished for you: mean squared error of relu(relu(X W_in) W_h) W_out, with gradients."""
    h1 = np.maximum(X @ params["W_in"], 0)
    h2 = np.maximum(h1 @ params["W_h"], 0)
    err = h2 @ params["W_out"] - Y
    loss = float(np.mean(err**2))
    d_out = 2 * err / err.size
    g_out = h2.T @ d_out
    d_h2 = (d_out @ params["W_out"].T) * (h2 > 0)
    g_h = h1.T @ d_h2
    d_h1 = (d_h2 @ params["W_h"].T) * (h1 > 0)
    g_in = X.T @ d_h1
    return loss, {"W_in": g_in, "W_h": g_h, "W_out": g_out}


def adamw_update(p, g, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Finished for you: one AdamW step without weight decay. `state` is a dict that you keep per parameter."""
    t = state.get("t", 0) + 1
    m = beta1 * state.get("m", np.zeros_like(p)) + (1 - beta1) * g
    v = beta2 * state.get("v", np.zeros_like(p)) + (1 - beta2) * g * g
    state.update(t=t, m=m, v=v)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps)


def train(use_muon, steps=60, lr_muon=0.02, lr_adam=0.01, mu=0.95):
    """Return {"losses": [...], "deltas": {name: first-step change}}."""
    X, Y, params = make_problem()
    start = {k: v.copy() for k, v in params.items()}
    states = {k: {} for k in params}
    B = np.zeros_like(params["W_h"])
    losses, deltas = [], {}
    for step in range(steps):
        loss, grads = loss_and_grads(params, X, Y)
        losses.append(loss)
        new = {}
        for name in ("W_in", "W_out"):
            new[name] = adamw_update(params[name], grads[name], states[name], lr_adam)
        if use_muon:
            B = mu * B + grads["W_h"]
            new["W_h"] = params["W_h"] - lr_muon * newton_schulz(B)
        else:
            new["W_h"] = adamw_update(params["W_h"], grads["W_h"], states["W_h"], lr_adam)
        if step == 0:
            deltas = {k: new[k] - start[k] for k in new}
        params = new
    return {"losses": losses, "deltas": deltas}
