from math import comb


def layout(N, K, m=1, Ks=0):
    """Return the expert layout of a DeepSeekMoE layer as a dict.

    Keys: n_routed, k_active, expert_size, total_ffn, active_ffn, combinations.
    """
    n_routed = m * N - Ks
    k_active = m * K - Ks
    size = 1.0 / m
    return {
        "n_routed": n_routed,
        "k_active": k_active,
        "expert_size": size,
        "total_ffn": (n_routed + Ks) * size,
        "active_ffn": (k_active + Ks) * size,
        "combinations": comb(n_routed, k_active),
    }
