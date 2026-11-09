"""Bias and variance of GAE(gamma, lambda) on a small random MDP.

The experiment fixes a start state s_0 and first action a_0, then estimates
A_{pi,gamma}(s_0, a_0) from sampled trajectories with GAE (Equation 16). The true
advantage comes from an exact solve. Two value functions are compared: the exact
V_{pi,gamma}, and V_{pi,gamma} plus a fixed error per state.

The estimate is A_hat = Q_hat - V(s_0), and the paper's Proposition 1 says the
baseline V(s_0) does not bias the policy gradient. So the bias that matters is the
bias of Q_hat = A_hat + V(s_0) against Q_{pi,gamma}(s_0, a_0), and that is what the
table reports.

Run: python3 scripts/bias_variance.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

S, A_N = 5, 2
gamma = 0.95
T = 120  # trajectory length; gamma^T is about 2e-3, and V(s_T) is bootstrapped anyway
N = 20000  # sampled trajectories
SIGMA = 0.3  # reward noise
ERR_SCALE = 1.0  # size of the per-state value error
S0, A0 = 0, 0

rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(S) * 0.5, size=(S, A_N))  # P[s, a, s']
r_mean = rng.normal(size=(S, A_N))  # E[r | s, a]
pi = rng.dirichlet(np.ones(A_N), size=S)  # pi[s, a]

P_pi = np.einsum("sa,sat->st", pi, P)
r_pi = np.sum(pi * r_mean, axis=1)
V_true = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
Q_true = r_mean + gamma * np.einsum("sat,t->sa", P, V_true)
A_true = Q_true[S0, A0] - V_true[S0]
print(f"true advantage A(s_0, a_0) = {A_true:.4f}")

err = ERR_SCALE * np.random.default_rng(1).normal(size=S)
V_bad = V_true + err
print("value error per state (V_bad - V_true):", err)


def rollouts(seed):
    """Return states (N, T+1), rewards (N, T) for N trajectories that start with (s_0, a_0)."""
    g = np.random.default_rng(seed)
    s = np.full(N, S0)
    a = np.full(N, A0)
    states = [s]
    rewards = []
    for t in range(T):
        rewards.append(r_mean[s, a] + SIGMA * g.normal(size=N))
        cum = P[s, a].cumsum(axis=1)
        s = (g.random(N)[:, None] > cum).sum(axis=1).clip(max=S - 1)
        states.append(s)
        cum = pi[s].cumsum(axis=1)
        a = (g.random(N)[:, None] > cum).sum(axis=1).clip(max=A_N - 1)
    return np.stack(states, axis=1), np.stack(rewards, axis=1)


states, rewards = rollouts(2)


def gae_at_zero(V, lam):
    """GAE(gamma, lambda) at t = 0 for every trajectory: sum_l (gamma lambda)^l delta_l (Equation 16)."""
    v = V[states]  # (N, T+1)
    delta = rewards + gamma * v[:, 1:] - v[:, :-1]
    w = (gamma * lam) ** np.arange(T)
    return delta @ w


def analytic_bias(err, lam):
    """E[A_hat + V(s_0)] - Q(s_0, a_0) for V = V_true + err.

    From E[delta_l] = gamma E e(s_{l+1}) - E e(s_l) for l >= 0 (Equation 10 with an error e),
    plus e(s_0) because V(s_0) is added back.
    """
    d = np.zeros(S)
    d[S0] = 1.0
    d_next = P[S0, A0]  # distribution of s_1 given (s_0, a_0)
    total = err[S0]
    for l in range(T):
        total += (gamma * lam) ** l * (gamma * (d_next @ err) - d @ err)
        d, d_next = d_next, d_next @ P_pi
    return total


lams = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
print()
print(f"{N} trajectories of length {T}, gamma = {gamma}, reward noise sd = {SIGMA}")
print()
print("| lambda | exact V: bias | exact V: std | wrong V: bias | wrong V: analytic bias | wrong V: std |")
print("|---|---|---|---|---|---|")
for lam in lams:
    ex = gae_at_zero(V_true, lam)
    bd = gae_at_zero(V_bad, lam)
    print(
        f"| {lam} | {ex.mean() - A_true:+.3f} | {ex.std():.3f} | {bd.mean() + V_bad[S0] - Q_true[S0, A0]:+.3f} "
        f"| {analytic_bias(err, lam):+.3f} | {bd.std():.3f} |"
    )
print(f"\nstandard error of each bias entry is about std / sqrt(N) = {gae_at_zero(V_true, 1.0).std() / np.sqrt(N):.3f} at lambda = 1")

rms = {
    lam: np.sqrt((gae_at_zero(V_bad, lam).mean() + V_bad[S0] - Q_true[S0, A0]) ** 2 + gae_at_zero(V_bad, lam).var())
    for lam in np.linspace(0, 1, 21)
}
best = min(rms, key=rms.get)
print(f"root mean squared error with the wrong V is smallest at lambda = {best:.2f} ({rms[best]:.3f}); at 0 it is {rms[0.0]:.3f}, at 1 it is {rms[1.0]:.3f}")
