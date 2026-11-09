# Table 5 and Table 6 of the paper (Appendix A). Compute is in PF-days.
ALPHA_CMIN, CMIN_C = 0.050, 3.1e8  # Equation 1.3
N_E, P_N = 1.3e9, 0.73  # N_opt = N_E * C ** P_N
D_E, P_D = 2e10, 0.27  # D_opt = D_E * C ** P_D, one epoch
B_STAR, ALPHA_B = 2.1e8, 0.21  # Equation 1.4


def optimal_loss(c):
    """L(C_min) = (C_c / C_min) ** alpha_C, Equation 1.3."""
    raise NotImplementedError


def compute_for_loss(loss):
    """Invert Equation 1.3: the C_min in PF-days that reaches the given loss."""
    raise NotImplementedError


def optimal_model_size(c):
    """N_opt = N_e * C_min ** 0.73 non-embedding parameters."""
    raise NotImplementedError


def optimal_tokens(c):
    """D_opt = D_e * C_min ** 0.27 tokens for a single epoch."""
    raise NotImplementedError


def critical_batch_size(loss):
    """B_crit(L) = B_star / L ** (1 / alpha_B) tokens, Equation 1.4."""
    raise NotImplementedError
