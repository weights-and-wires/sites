"""Euler sampling with classifier-free guidance on a 2-D toy with a known flow
(Section 2.1 Equation 3, Section 2.2 Equation 4, Section 3.2 Equation 7).

Data given condition c is N(mu_c, S^2 I). With the OT path
x_t = (1 - t) x0 + t x1, the marginal field has a closed form:
    var = (1 - t)^2 + t^2 S^2
    u_c(x, t) = mu_c + (t S^2 - (1 - t)) / var * (x - t mu_c)
The unconditional field is the mixture of the two components' fields, weighted
by the posterior of each component given x at time t.

Run: python3 scripts/toy_cfg.py
"""

import numpy as np

S = 0.5
MU = np.array([[-2.0, 0.0], [2.0, 0.0]])


def v_cond(x, t, c):
    var = (1 - t) ** 2 + t**2 * S**2
    return MU[c] + (t * S**2 - (1 - t)) / var * (x - t * MU[c])


def v_uncond(x, t):
    var = (1 - t) ** 2 + t**2 * S**2
    logw = np.stack([-((x - t * MU[c]) ** 2).sum(-1) / (2 * var) for c in range(2)])
    w = np.exp(logw - logw.max(0))
    w /= w.sum(0)
    return sum(w[c][..., None] * v_cond(x, t, c) for c in range(2))


def sample(x0, ts, c, w):
    x = x0.copy()
    for a, b in zip(ts[:-1], ts[1:]):
        vc = v_cond(x, a, c)
        vu = v_uncond(x, a)
        x = x + (b - a) * (vc + w * (vc - vu))
    return x


def sway(u, s):
    return u + s * (np.cos(np.pi / 2 * u) - 1 + u)


rng = np.random.default_rng(0)
x0 = rng.normal(size=(4000, 2))
target_c = 1  # the cluster centered at (+2, 0)

print("Data: N(mu_c, 0.5^2 I), mu_0 = (-2, 0), mu_1 = (+2, 0); condition c = 1")
print("Exact data std per axis:", S)

print()
print("Reference: 2000 Euler steps, w = 0")
ref = sample(x0, np.linspace(0, 1, 2001), target_c, 0.0)
print("mean", np.round(ref.mean(0), 3), " std", np.round(ref.std(0), 3))

print()
print("Euler, uniform steps, w = 0 (no guidance); error = mean |x - reference|")
for n in (2, 4, 8, 16, 32):
    out = sample(x0, np.linspace(0, 1, n + 1), target_c, 0.0)
    print(f"NFE={n:<3} mean {np.round(out.mean(0), 3)}  std {np.round(out.std(0), 3)}  error {np.abs(out - ref).mean():.4f}")

print()
print("Guidance strength w (Equation 4), 16 uniform steps")
for w in (0.0, 1.0, 2.0, 4.0):
    out = sample(x0, np.linspace(0, 1, 17), target_c, w)
    print(f"w={w:<4} mean {np.round(out.mean(0), 3)}  std {np.round(out.std(0), 3)}  wrong-side fraction {(out[:, 0] < 0).mean():.4f}")

print()
print("Sway steps against uniform steps, w = 0, error against the 2000-step reference")
u = lambda n: np.linspace(0, 1, n + 1)
for n in (2, 4, 8):
    e_uni = np.abs(sample(x0, u(n), target_c, 0.0) - ref).mean()
    e_sway = np.abs(sample(x0, sway(u(n), -1.0), target_c, 0.0) - ref).mean()
    print(f"NFE={n:<3} uniform {e_uni:.4f}   sway s=-1 {e_sway:.4f}")

print()
print("Check by hand at t = 0, x = (0.5, 0), where the conditional field is mu_c - x:")
print("v_cond(c=1) =", v_cond(np.array([0.5, 0.0]), 0.0, 1), " v_cond(c=0) =", v_cond(np.array([0.5, 0.0]), 0.0, 0))
print("v_uncond    =", v_uncond(np.array([0.5, 0.0]), 0.0))

vc, vu = v_cond(np.array([0.5, 0.0]), 0.0, 1), v_uncond(np.array([0.5, 0.0]), 0.0)
print("v_CFG at w = 2:", vc + 2.0 * (vc - vu), " at w = 1:", vc + 1.0 * (vc - vu))
