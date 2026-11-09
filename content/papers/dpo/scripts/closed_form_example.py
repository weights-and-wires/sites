"""Compute the numbers for the closed-form optimal policy example (Equations 4, 5, 6).

Run: python3 content/papers/dpo/scripts/closed_form_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

ref = np.array([0.70, 0.25, 0.05])  # pi_ref(y | x) over three responses
r = np.array([0.0, 1.0, 2.0])  # a reward r(x, y)


def optimal_policy(ref, r, beta):
    w = ref * np.exp(r / beta)
    Z = w.sum()
    return w / Z, Z


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


for beta in (1.0, 0.5):
    pi, Z = optimal_policy(ref, r, beta)
    print(f"beta = {beta}")
    print("  unnormalized weights ref * exp(r/beta) =", (ref * np.exp(r / beta)).tolist())
    print("  Z(x)                                   =", round(float(Z), 4))
    print("  pi*                                    =", pi.tolist())
    print("  sum of pi*                             =", round(float(pi.sum()), 6))
    # Equation 5: r = beta * log(pi*/ref) + beta * log Z
    back = beta * np.log(pi / ref) + beta * np.log(Z)
    print("  beta log(pi*/ref)                      =", (beta * np.log(pi / ref)).tolist())
    print("  beta log(pi*/ref) + beta log Z         =", back.tolist(), "(equals r)")

print()
print("Z cancels in the Bradley-Terry probability (beta = 1, compare responses 2 and 0)")
beta = 1.0
pi, Z = optimal_policy(ref, r, beta)
p_from_reward = sigmoid(r[2] - r[0])
p_from_policy = sigmoid(beta * np.log(pi[2] / ref[2]) - beta * np.log(pi[0] / ref[0]))  # Equation 6
print("  p(y2 > y0) from the rewards      =", round(float(p_from_reward), 4))
print("  p(y2 > y0) from pi* and pi_ref   =", round(float(p_from_policy), 4))

print()
print("Adding a constant f(x) to the reward leaves pi* unchanged (Lemma 2)")
pi_shift, Z_shift = optimal_policy(ref, r + 5.0, beta)
print("  pi* for r          =", pi.tolist(), " Z =", round(float(Z), 4))
print("  pi* for r + 5      =", pi_shift.tolist(), " Z =", round(float(Z_shift), 4))
print("  Z ratio exp(5/beta) =", round(float(np.exp(5.0 / beta)), 4))
print("  p(y2 > y0) for r + 5 =", round(float(sigmoid((r[2] + 5) - (r[0] + 5))), 4))

print()
print("Self-check: pi_ref = [0.5, 0.5], r = [0, 1], beta = 1")
pi2, Z2 = optimal_policy(np.array([0.5, 0.5]), np.array([0.0, 1.0]), 1.0)
print("pi*                          =", pi2.tolist())
print("sigmoid(1)                   =", round(float(sigmoid(1.0)), 4))
print("Z                            =", round(float(Z2), 4))
