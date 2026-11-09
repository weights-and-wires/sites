"""Multi-token prediction alignment and speculative-decoding arithmetic (Section 2.2, Equations 21-25, Section 5.4.3)."""
import numpy as np


def mtp_pairs(T, k):
    """1-indexed (position i, target token index) pairs for depth k.

    Equation 22-23: h^k_i for i = 1..T-k predicts token t_{i+k+1}.
    """
    return [(i, i + k + 1) for i in range(1, T - k + 1)]


def main():
    T = 6  # sequence length; tokens t_1 .. t_{T+1}
    print(f"T = {T}: tokens t_1..t_{T + 1}")
    for k in (0, 1, 2):
        pairs = mtp_pairs(T, k)
        label = "main model" if k == 0 else f"MTP depth {k}"
        print(f"  {label}: {len(pairs)} positions; (i -> target):", ", ".join(f"{i}->t{j}" for i, j in pairs))
    # Equation 24 sums i = 2+k .. T+1 and divides by T (not by the number of terms).
    for k in (1, 2):
        terms = T + 1 - (2 + k) + 1
        print(f"  Equation 24, k = {k}: {terms} terms, divided by T = {T}")

    # With uniform predictions over V tokens, every log P is -log V.
    V = 100
    print(f"\nuniform predictions, V = {V}:")
    for k in (1, 2):
        terms = T - k
        print(f"  depth {k}: L = {terms}/{T} * ln V = {terms / T * np.log(V):.4f}")
    lam = 0.3
    D = 2
    l1, l2 = 2.0, 3.0
    print(f"\nEquation 25: lambda = {lam}, D = {D}, L^1 = {l1}, L^2 = {l2} -> L_MTP = {lam / D * (l1 + l2):.4f}")

    # Equation 21 shapes
    d = 4
    print(f"\nEquation 21 shapes with d = {d}: M_k is {d} x {2 * d}, input is [h; emb] of length {2 * d}, output length {d}")

    # Speculative decoding: one verified token plus the extra token when accepted.
    print("\nspeculative decoding with D = 1 (Section 5.4.3: acceptance 85-90%, 1.8x TPS reported):")
    for p in (0.85, 0.90):
        print(f"  acceptance {p:.2f}: expected tokens per decoding step = {1 + p:.2f}")


if __name__ == "__main__":
    main()
