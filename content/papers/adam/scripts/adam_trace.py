"""Trace three Adam steps on one scalar parameter (Algorithm 1, Section 2).

Run: python3 scripts/adam_trace.py
"""

import numpy as np

alpha, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8
grads = [2.0, -1.0, 1.0]

m = v = 0.0
print(f"alpha={alpha}, beta1={beta1}, beta2={beta2}, eps={eps}")
for t, g in enumerate(grads, start=1):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    delta = alpha * m_hat / (np.sqrt(v_hat) + eps)
    print(
        f"t={t}: g={g:+.1f}  m={m:+.4f}  v={v:.6f}  "
        f"m_hat={m_hat:+.4f}  v_hat={v_hat:.4f}  step/alpha={delta / alpha:+.4f}"
    )
