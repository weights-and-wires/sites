"""Compute the ratios quoted on the page from numbers the paper reports.

Run: python3 content/papers/instructgpt/scripts/pipeline_numbers.py
"""

# Section 1, Figure 1.
print(f"175B / 1.3B parameters = {175 / 1.3:.1f}")

# Section 5.1, lesson 1: compute in petaflops/s-days.
sft, ppo_ptx, gpt3 = 4.9, 60.0, 3640.0
print(f"SFT + PPO-ptx = {sft + ppo_ptx:.1f} petaflops/s-days")
print(f"as a share of GPT-3 pretraining ({gpt3:.0f}) = {100 * (sft + ppo_ptx) / gpt3:.2f}%")

# Appendix A.3, Table 6: prompts per stage (train split).
sft_train = 11295 + 1430
rm_train = 6623 + 26584
ppo_train = 31144
print(f"SFT train prompts = {sft_train}, RM train prompts = {rm_train}, PPO train prompts = {ppo_train}")
print(f"RM / SFT = {rm_train / sft_train:.2f}")

# Appendix C.4: RL training.
episodes, batch, minibatch = 256_000, 512, 64
print(f"iterations = {episodes // batch}, minibatches per batch = {batch // minibatch}")
print(f"PPO-ptx pretraining examples = 8 x {episodes} = {8 * episodes}")
