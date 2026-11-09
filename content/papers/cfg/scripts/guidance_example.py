"""Compute the numbers for the worked examples on the cfg page.

Part A: the log-SNR parameterization and the distribution of lambda (Section 2).
Part B: the guided score of Equation 6 at one point.
Part C: the implicit classifier identity (Section 3.2).
Part D: the guided density on a grid and the Algorithm 2 sampler, in 1-D.

The data are two classes, x | c ~ N(mu_c, 1) with mu = (-2, +2) and equal
class probabilities, so every score has a closed form.

Run: python3 scripts/guidance_example.py
"""

import numpy as np

MU = np.array([-2.0, 2.0])
S = 1.0


def alpha_sigma(lam):
    a2 = 1.0 / (1.0 + np.exp(-lam))
    return np.sqrt(a2), np.sqrt(1.0 - a2)


def eps_cond(z, c, lam):
    a, sg = alpha_sigma(lam)
    return sg * (z - a * MU[c]) / (a**2 * S**2 + sg**2)


def log_p_cond(z, c, lam):
    a, sg = alpha_sigma(lam)
    v = a**2 * S**2 + sg**2
    return -0.5 * (z - a * MU[c]) ** 2 / v - 0.5 * np.log(2 * np.pi * v)


def log_p(z, lam):
    return np.logaddexp(log_p_cond(z, 0, lam), log_p_cond(z, 1, lam)) + np.log(0.5)


def eps_uncond(z, lam):
    a, sg = alpha_sigma(lam)
    v = a**2 * S**2 + sg**2
    r1 = 1.0 / (1.0 + np.exp(log_p_cond(z, 0, lam) - log_p_cond(z, 1, lam)))
    mean = a * (r1 * MU[1] + (1 - r1) * MU[0])
    return sg * (z - mean) / v


def lambdas(T, lam_min=-20.0, lam_max=20.0):
    b = np.arctan(np.exp(-lam_max / 2))
    a = np.arctan(np.exp(-lam_min / 2)) - b
    u = np.linspace(1.0, 0.0, T)
    return -2.0 * np.log(np.tan(a * u + b))


def sample(c, w, n, T, rng, v=0.0):
    lam = lambdas(T)
    z = rng.standard_normal(n)
    for t in range(T):
        a, sg = alpha_sigma(lam[t])
        eps = (1 + w) * eps_cond(z, c, lam[t]) - w * eps_uncond(z, lam[t])
        x = (z - sg * eps) / a
        if t == T - 1:
            return x
        a2, sg2 = alpha_sigma(lam[t + 1])
        r = np.exp(lam[t] - lam[t + 1])
        mean = r * (a2 / a) * z + (1 - r) * a2 * x
        var = ((1 - r) * sg2**2) ** (1 - v) * ((1 - r) * sg**2) ** v
        z = mean + np.sqrt(var) * rng.standard_normal(n)


print("Part A: alpha_lambda and sigma_lambda")
for lam in (-20.0, -4.0, 0.0, 4.0, 20.0):
    a, sg = alpha_sigma(lam)
    print(f"  lambda={lam:6.1f}  alpha^2={a**2:.6f}  sigma^2={sg**2:.6f}  alpha={a:.4f}  sigma={sg:.4f}")
lam8 = lambdas(9)
print("  lambda at 9 uniform steps in u (increasing):", np.round(lam8, 3).tolist())
print("  lambda for T=256, first/second/middle/last:", [round(float(lambdas(256)[i]), 3) for i in (0, 1, 128, 255)])

betas = np.linspace(1e-4, 0.02, 1000)
ab = np.cumprod(1 - betas)
print("  DDPM schedule (T=1000) as log-SNR: lambda = log(alpha_bar / (1 - alpha_bar))")
for t in (1, 100, 500, 1000):
    print(f"    t={t:<5} alpha_bar={ab[t - 1]:.6f}  lambda={np.log(ab[t - 1] / (1 - ab[t - 1])):8.3f}")

print("Part B: Equation 6 at lambda=0, z=1, class c=1")
z, lam, c = 1.0, 0.0, 1
ec, eu = eps_cond(z, c, lam), eps_uncond(z, lam)
print(f"  eps(z,c)={ec:.4f}  eps(z)={eu:.4f}  difference eps(z,c)-eps(z)={ec - eu:.4f}")
for w in (0.0, 1.0, 2.0, 4.0):
    print(f"  w={w:3.1f}  (1+w) eps_c - w eps_u = {(1 + w) * ec - w * eu:.4f}")
print(f"  (a sampler that used w*eps_c + (1-w)*eps_u at w=2 would get {2 * ec - eu:.4f})")

print("Part C: implicit classifier, grad log p(c|z) = -(1/sigma)(eps_c - eps_u)")
a, sg = alpha_sigma(lam)
h = 1e-5
post = lambda zz: log_p_cond(zz, c, lam) + np.log(0.5) - log_p(zz, lam)
fd = (post(z + h) - post(z - h)) / (2 * h)
print(f"  finite difference of log p(c=1|z) = {fd:.4f}")
print(f"  -(1/sigma)(eps_c - eps_u)         = {-(ec - eu) / sg:.4f}")
print(f"  p(c=1|z) = {np.exp(post(z)):.4f}")

print("Part D: guided density on a grid (clean data, lambda -> infinity) versus the sampler")
x = np.linspace(-12, 12, 48001)
dx = x[1] - x[0]
lc = -0.5 * (x - MU[1]) ** 2 / S**2 - 0.5 * np.log(2 * np.pi * S**2)
l0 = -0.5 * (x - MU[0]) ** 2 / S**2 - 0.5 * np.log(2 * np.pi * S**2)
lm = np.logaddexp(lc, l0) + np.log(0.5)
rng = np.random.default_rng(0)
print("  w     grid_mean grid_std | sampler_mean sampler_std")
for w in (0.0, 1.0, 2.0, 4.0):
    lg = (1 + w) * lc - w * lm
    p = np.exp(lg - lg.max())
    p /= p.sum() * dx
    m = np.sum(x * p) * dx
    sd = np.sqrt(np.sum((x - m) ** 2 * p) * dx)
    xs = sample(1, w, 100000, 256, rng)
    print(f"  {w:<5} {m:9.4f} {sd:8.4f} | {xs.mean():12.4f} {xs.std():11.4f}")
