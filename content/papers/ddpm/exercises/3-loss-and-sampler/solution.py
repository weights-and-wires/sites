import numpy as np


def make_schedule(T, beta_1=1e-4, beta_T=0.02):
    """Finished for you. Index i holds timestep t = i + 1."""
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1.0 - betas
    return betas, alphas, np.cumprod(alphas)


def q_sample(x0, t, eps, alpha_bars):
    """Finished for you. Equation 4. t is an int or an int array of shape (batch,)."""
    ab = alpha_bars[np.asarray(t) - 1]
    ab = np.reshape(ab, (-1,) + (1,) * (np.ndim(x0) - 1))
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps


def ddpm_loss(eps_model, x0, t, eps, alpha_bars):
    """One evaluation of the loss in Algorithm 1, line 5 (Equation 14).

    x0, eps: (batch, dim). t: int array (batch,), values in 1..T.
    eps_model(x_t, t) returns an array shaped like x_t.
    Returns a float: squared error summed over dim, averaged over the batch.
    """
    x_t = q_sample(x0, t, eps, alpha_bars)
    err = eps - eps_model(x_t, t)
    return float(np.mean(np.sum(err**2, axis=-1)))


def sample(eps_model, shape, betas, rng, sigma="beta"):
    """Algorithm 2. Returns x_0 with the given shape (batch, dim).

    rng is a numpy Generator. sigma is "beta" or "beta_tilde".
    """
    T = len(betas)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    x = rng.standard_normal(shape)
    for t in range(T, 0, -1):
        ab_prev = 1.0 if t == 1 else alpha_bars[t - 2]
        if sigma == "beta":
            var = betas[t - 1]
        else:
            var = (1 - ab_prev) / (1 - alpha_bars[t - 1]) * betas[t - 1]
        z = rng.standard_normal(shape) if t > 1 else 0.0
        eps_hat = eps_model(x, t)
        x = (x - (1 - alphas[t - 1]) / np.sqrt(1 - alpha_bars[t - 1]) * eps_hat) / np.sqrt(alphas[t - 1]) + np.sqrt(var) * z
    return x
