"""Count the parameters and multiply-adds of the alignment model (Appendix A.1.2, A.2.3).

Run: python3 scripts/alignment_cost.py
"""

n = 1000  # hidden units (Appendix A.2.3)
n_prime = 1000  # alignment hidden units (Appendix A.2.3)
Tx = Ty = 50

params = n_prime * n + n_prime * 2 * n + n_prime  # W_a, U_a, v_a
print("alignment parameters:", params)

pairs = Tx * Ty
print("alignment evaluations per sentence pair:", pairs)

# Multiply-adds. The tanh and the additions are counted separately and are small.
naive = pairs * (n_prime * n + n_prime * 2 * n + n_prime)
precomputed = Ty * (n_prime * n) + Tx * (n_prime * 2 * n) + pairs * n_prime
print("multiply-adds, recomputing everything per pair:", naive)
print("multiply-adds, precomputing U_a h_j and W_a s_{i-1}:", precomputed)
print("ratio:", round(naive / precomputed, 1))
