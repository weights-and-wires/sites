from math import comb


def layout(N, K, m=1, Ks=0):
    """Return the expert layout of a DeepSeekMoE layer as a dict.

    Keys: n_routed, k_active, expert_size, total_ffn, active_ffn, combinations.
    """
    raise NotImplementedError
