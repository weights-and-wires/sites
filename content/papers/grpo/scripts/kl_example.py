"""The KL estimator of Equation 4, compared with the one inside PPO's reward (Equation 2)."""

import numpy as np

from grpo_lib import kl_k3

print("One token: pi = 0.5, pi_ref = 0.25")
lp, lr_ = np.log(0.5), np.log(0.25)
print("  k3 (Equation 4)            ", round(float(kl_k3(lp, lr_)), 4))
print("  log(pi / pi_ref) (Equation 2)", round(float(lp - lr_), 4))
print("  the other direction, pi = 0.25, pi_ref = 0.5:", round(float(kl_k3(lr_, lp)), 4))
print("  k3 at pi = pi_ref:", float(kl_k3(lp, lp)))

print()
print("Exact expectation over the policy of a 3-token vocabulary")
p = np.array([0.6, 0.3, 0.1])
q = np.array([0.2, 0.5, 0.3])
exact = float(np.sum(p * np.log(p / q)))
k3 = float(np.sum(p * kl_k3(np.log(p), np.log(q))))
k1 = float(np.sum(p * (np.log(p) - np.log(q))))
print("  KL(pi || pi_ref)              ", round(exact, 4))
print("  E_pi[k3]                      ", round(k3, 4))
print("  E_pi[log pi - log pi_ref] (k1)", round(k1, 4))
print("  values of k3 per token        ", np.round(kl_k3(np.log(p), np.log(q)), 4))
print("  values of k1 per token        ", np.round(np.log(p / q), 4))

print()
print("Monte Carlo, 100000 samples, seed 0")
rng = np.random.default_rng(0)
a = rng.choice(3, size=100_000, p=p)
s3 = kl_k3(np.log(p[a]), np.log(q[a]))
s1 = np.log(p[a] / q[a])
print("  mean k3", round(float(s3.mean()), 4), " std", round(float(s3.std()), 4), " min", round(float(s3.min()), 4))
print("  mean k1", round(float(s1.mean()), 4), " std", round(float(s1.std()), 4), " min", round(float(s1.min()), 4))
print("  share of k1 samples below 0:", round(float((s1 < 0).mean()), 4))
