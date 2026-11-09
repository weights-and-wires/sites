"""Compute the numbers for the learning-rate schedule (Section 5.3, Equation 3).

Run: python3 scripts/lr_schedule.py
"""


def lrate(step, d_model=512, warmup=4000):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


for step in (1, 1000, 4000, 8000, 16000, 100000):
    print(f"step {step:>6}: lrate = {lrate(step):.3e}")
print()
print("peak at step 4000:", f"{lrate(4000):.3e}")
print("d_model = 256 peak:", f"{lrate(4000, d_model=256):.3e}")
