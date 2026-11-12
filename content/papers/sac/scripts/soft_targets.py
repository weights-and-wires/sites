"""Numbers for the "Soft targets" example (Section 4.2 Equations 5 to 9, Algorithm 1).

Run: python3 scripts/soft_targets.py
"""

import numpy as np

gamma = 0.99

# A minibatch of three transitions. The actions a_i were sampled from the current policy,
# so logp_i = log pi(a_i | s_i), and q1, q2 are the two Q critics at (s_i, a_i).
q1 = np.array([1.0, 0.5, 2.0])
q2 = np.array([0.8, 0.9, 1.5])
logp = np.array([-1.2, -0.3, -0.5])
v = np.array([1.5, 0.4, 1.9])  # V_psi(s_i)

q_min = np.minimum(q1, q2)
v_target = q_min - logp
print(f"min(Q1, Q2) = {q_min}")
print(f"V target = min(Q1,Q2) - log pi = {v_target}")
print(f"J_V (Eq 5) = {np.mean(0.5 * (v - v_target) ** 2):.5f}")
print(f"dJ_V / dV = V - target = {v - v_target}")

# Q target (Equation 8) from the target value network at the next state.
r = np.array([1.0, 0.0, 2.0])
v_bar_next = np.array([3.0, 2.0, 1.0])
done = np.array([0.0, 0.0, 1.0])
q_hat = r + gamma * (1 - done) * v_bar_next
print(f"Q target = r + gamma (1 - d) V_bar(s') = {q_hat}")
print(f"without the mask the third target would be {r[2] + gamma * v_bar_next[2]:.2f}")
for name, q in (("Q1", q1), ("Q2", q2)):
    print(f"J_Q for {name} (Eq 7) = {np.mean(0.5 * (q - q_hat) ** 2):.5f}")

# Why the minimum: noisy Q estimates of five actions that are all worth 0.
rng = np.random.default_rng(0)
n, trials, noise = 5, 200000, 1.0
e1 = rng.normal(0, noise, (trials, n))
e2 = rng.normal(0, noise, (trials, n))
print(f"true value of every action = 0, noise std = {noise}, {n} actions")
print(f"  E[max_a Q1(a)]            = {e1.max(axis=1).mean():.3f}")
print(f"  E[max_a min(Q1, Q2)(a)]   = {np.minimum(e1, e2).max(axis=1).mean():.3f}")
pick = e1.argmax(axis=1)
print(f"  E[Q2 at argmax of Q1]     = {e2[np.arange(trials), pick].mean():.3f}")

# Soft version: V = alpha * logsumexp(Q / alpha) on the same noisy Q, alpha = 1.
def lse(x):
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).ravel()

print(f"soft value, single Q:  E[logsumexp Q1]         = {lse(e1).mean():.3f}  (true value log 5 = {np.log(5):.3f})")
print(f"soft value, twin min:  E[logsumexp min(Q1,Q2)] = {lse(np.minimum(e1, e2)).mean():.3f}")
