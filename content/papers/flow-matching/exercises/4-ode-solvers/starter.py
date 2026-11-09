import numpy as np


def euler_sample(v, x0, n_steps):
    """Integrate dx/dt = v(x, t) from t = 0 to t = 1 with n_steps Euler steps."""
    raise NotImplementedError


def midpoint_sample(v, x0, n_steps):
    """Integrate dx/dt = v(x, t) from t = 0 to t = 1 with n_steps midpoint steps."""
    raise NotImplementedError


def gaussian_field(m, s, sigma_min=0.0):
    """Exact marginal field for a data distribution N(m, s^2) and the OT path.

    The marginal path is p_t = N(t m, Sigma_t^2) with
    Sigma_t^2 = (1 - (1 - sigma_min) t)^2 + t^2 s^2, and the field is
    u_t(x) = m + (Sigma_t' / Sigma_t) (x - t m).
    """

    def v(x, t):
        sig = 1.0 - (1.0 - sigma_min) * t
        big = np.sqrt(sig**2 + (t * s) ** 2)
        dbig = (-(1.0 - sigma_min) * sig + t * s**2) / big
        return m + dbig / big * (x - t * m)

    return v


def exact_endpoint(x0, m, s, sigma_min=0.0):
    """The exact value of phi_1(x0) for gaussian_field: m + sqrt(sigma_min^2 + s^2) x0."""
    return m + np.sqrt(sigma_min**2 + s**2) * x0
