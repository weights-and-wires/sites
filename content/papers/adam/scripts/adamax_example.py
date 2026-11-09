"""AdaMax on a short gradient sequence (Algorithm 2 and Equation 12, Section 7.1).

Run: python3 scripts/adamax_example.py
"""

alpha, beta1, beta2 = 0.002, 0.9, 0.999
grads = [2.0, -1.0, 0.5, 4.0]

m = u = 0.0
for t, g in enumerate(grads, start=1):
    m = beta1 * m + (1 - beta1) * g
    u = max(beta2 * u, abs(g))
    delta = (alpha / (1 - beta1**t)) * m / u
    print(f"t={t}: g={g:+.1f}  m={m:+.4f}  u={u:.4f}  step/alpha={delta / alpha:+.4f}")
