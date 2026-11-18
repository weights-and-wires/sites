"""Check the two scaling rules in Section 3.1 against Table 1 and Table 2.

Equation 10 defines the AdamW timescale tau = 1 / (eta * lambda * T), with T
the number of steps. The paper says tau * sqrt(TPP) lies between 2.6 and 2.9
for the 3.7B, 7B, 32B, and 36B-A4B models, where TPP is tokens per active
parameter. Equation 11 says the best peak learning rate scales as N^(-1/2);
the paper says eta * sqrt(N) lies between 6.8e-4 and 8.6e-4. N is in billions
of active backbone parameters, which is the unit that reproduces that range.
"""
import math

# name: (active backbone params in B, tokens, steps, peak LR, weight decay)
MODELS = {
    "0.9B": (0.881, 5.03e12, 600_000, 8e-4, 0.10),
    "3.7B": (3.775, 21.92e12, 1_100_000, 4e-4, 0.06),
    "7B": (6.946, 21.92e12, 1_100_000, 3e-4, 0.06),
    "32B": (32.213, 21.92e12, 1_100_000, 1.5e-4, 0.06),
    "36B-A4B": (4.665, 21.92e12, 1_100_000, 4e-4, 0.06),
    "375B-A23B": (23.590, None, None, 1.4e-4, 0.073),
}

print(f"{'model':<10} {'tau':>8} {'TPP':>8} {'tau*sqrt(TPP)':>14} {'eta*sqrt(N)':>12}")
for name, (n, tokens, steps, eta, wd) in MODELS.items():
    lr_rule = eta * math.sqrt(n)
    if steps is None:
        print(f"{name:<10} {'-':>8} {'-':>8} {'-':>14} {lr_rule:12.2e}")
        continue
    tau = 1 / (eta * wd * steps)
    tpp = tokens / (n * 1e9)
    print(f"{name:<10} {tau:8.4f} {tpp:8.0f} {tau * math.sqrt(tpp):14.2f} {lr_rule:12.2e}")

# Transfer example: predict the 32B peak LR from the 7B one with Equation 11.
pred = 3e-4 * math.sqrt(6.946 / 32.213)
print(f"\n7B -> 32B predicted peak LR: {pred:.3e} (paper uses 1.5e-4)")
