import numpy as np

# Appendix K, Table 2.
A, B, E, ALPHA = 4.299e3, 1.806e4, 2.7648, 0.4965
GAMMA = {"w": 2.6745, "a": 2.2102, "kv": 0.9578}
OFFSET = {"w": 0.3037, "a": 1.4072, "kv": 2.4185}


def eff_factor(p, gamma, offset=0.0):
    """Return 1 - exp(-p / gamma + offset), one factor of Equation 4."""
    p = np.asarray(p, dtype=float)
    return 1.0 - np.exp(-p / gamma + offset)


def n_eff(n, p_w, p_a, p_kv, offsets=False):
    """Return N_eff of Equation 4: n times the product of the three factors.

    With offsets=True, use the Table 2 offsets OFFSET; otherwise use 0.
    """
    off = OFFSET if offsets else {"w": 0.0, "a": 0.0, "kv": 0.0}
    return (
        np.asarray(n, dtype=float)
        * eff_factor(p_w, GAMMA["w"], off["w"])
        * eff_factor(p_a, GAMMA["a"], off["a"])
        * eff_factor(p_kv, GAMMA["kv"], off["kv"])
    )


def loss_hat(n, d, p_w, p_a, p_kv, offsets=False):
    """Return A * N_eff**-ALPHA + B * d**-ALPHA + E (Section 5, without PTQ)."""
    d = np.asarray(d, dtype=float)
    return A * n_eff(n, p_w, p_a, p_kv, offsets) ** -ALPHA + B * d**-ALPHA + E
