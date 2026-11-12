"""Numbers for the "Two evaluations per leaf" example.

Run: python3 scripts/leaf_mix_example.py
"""

# Main text, "Searching with policy and value networks": V = (1 - lam) v + lam z.
v, z = 0.3, -1.0
for lam in (0.0, 0.5, 1.0):
    print(f"lambda = {lam:.1f}: V = {(1 - lam) * v + lam * z:+.4f}")

# Methods, "Search algorithm": Q = (1 - lam) W_v / N_v + lam W_r / N_r.
# The two statistics are updated by separate backward passes, so N_v and N_r
# can differ at any moment.
lam = 0.5
Wv, Nv = 1.2, 4
Wr, Nr = -1.0, 3
q_v, q_r = Wv / Nv, Wr / Nr
print(f"W_v / N_v = {q_v:+.4f}, W_r / N_r = {q_r:+.4f}, Q = {(1 - lam) * q_v + lam * q_r:+.4f}")

# When every simulation contributes one value-network and one rollout result,
# the two forms are the same number.
vs = [0.3, 0.1, -0.5, 0.7]
zs = [-1.0, 1.0, -1.0, 1.0]
mean_of_mixed = sum((1 - lam) * a + lam * b for a, b in zip(vs, zs)) / 4
mixed_of_means = (1 - lam) * sum(vs) / 4 + lam * sum(zs) / 4
print(f"mean of V = {mean_of_mixed:+.4f}, mix of means = {mixed_of_means:+.4f}")
