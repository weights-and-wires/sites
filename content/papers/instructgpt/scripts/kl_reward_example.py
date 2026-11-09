"""Compute the per-token KL-penalized reward and the Equation 2 objective for a toy rollout.

Run: python3 content/papers/instructgpt/scripts/kl_reward_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# One response of 5 tokens. Log-probabilities of the sampled tokens under the RL
# policy and under the frozen SFT policy, for the same prompt and prefix.
logp_rl = np.array([-1.2, -0.8, -2.1, -0.5, -1.0])
logp_sft = np.array([-1.5, -1.0, -1.9, -0.9, -1.4])
rm_score = 1.8  # r_theta(x, y), one scalar for the whole response


def token_rewards(logp_rl, logp_sft, rm_score, beta):
    r = -beta * (logp_rl - logp_sft)
    r[-1] += rm_score
    return r


log_ratio = logp_rl - logp_sft
print("per-token log ratio log pi_RL - log pi_SFT =", np.round(log_ratio, 4).tolist())
print("sequence log ratio (sum)                   =", round(float(log_ratio.sum()), 4))

for beta in (0.02, 0.1, 0.5):
    r = token_rewards(logp_rl, logp_sft, rm_score, beta)
    total = r.sum()
    eq2 = rm_score - beta * np.log(np.exp(logp_rl.sum()) / np.exp(logp_sft.sum()))
    print(f"beta={beta}: per-token reward = {np.round(r, 4).tolist()}")
    print(f"          sum = {total:.4f}; Equation 2 integrand = {eq2:.4f}")

# The pretraining term of Equation 2.
beta, gamma = 0.02, 27.8
pre_logp = -2.3  # log pi_RL of one pretraining text, made up
integrand = rm_score - beta * log_ratio.sum()
print(f"PPO part for beta=0.02: {integrand:.4f}")
print(f"gamma * log-likelihood term with gamma=27.8: {gamma * pre_logp:.2f}")
print(f"combined objective for this one sample pair: {integrand + gamma * pre_logp:.4f}")
