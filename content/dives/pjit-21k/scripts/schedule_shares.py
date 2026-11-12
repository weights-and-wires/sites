"""Exact numbers for the logit-normal timestep schedules on the page.

t = sigmoid(z) with z ~ Normal(mu, s). P(a < t < b) = Phi((logit b - mu)/s) - Phi((logit a - mu)/s).
t = 1 is pure noise (the post's plot runs clean to noise, and its v-loss clamps t at 0.1).
"""
import math

def Phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def logit(p): return math.log(p / (1 - p))

def mass(mu, s, a, b):
    return Phi((logit(b) - mu) / s) - Phi((logit(a) - mu) / s)

def mean_t(mu, s, n=200001):
    # integrate sigmoid(z) * N(z; mu, s) on a fine grid over +-8 sd
    lo, hi = mu - 8 * s, mu + 8 * s
    h = (hi - lo) / (n - 1)
    tot = 0.0
    for i in range(n):
        z = lo + i * h
        tot += (1 / (1 + math.exp(-z))) * math.exp(-0.5 * ((z - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    return tot * h

P1, P2 = (0.8, 0.8), (-0.2, 1.0)
for name, (mu, s) in (("phase 1 LogitNormal(0.8, 0.8)", P1), ("phase 2 LogitNormal(-0.2, 1.0)", P2)):
    print(name)
    print("  P(0.08 < t < 0.13) = %.4f%%" % (100 * mass(mu, s, 0.08, 0.13)))
    print("  P(t < 0.1)         = %.4f%%" % (100 * mass(mu, s, 1e-12, 0.1)))
    print("  P(t > 0.9)         = %.4f%%" % (100 * mass(mu, s, 0.9, 1 - 1e-12)))
    print("  mean t             = %.3f" % mean_t(mu, s))
for a, b in ((0.08, 0.13), (0.075, 0.125)):
    p1, p2 = mass(*P1, a, b), mass(*P2, a, b)
    print("band [%.3f, %.3f]: phase 1 %.4f%%, phase 2 %.2f%%, ratio %.0fx" % (a, b, 100 * p1, 100 * p2, p2 / p1))
# The post labels its band "0.08 to 0.13" and reports 3.0% and 107x. The band
# [0.075, 0.125] (centered on t = 0.10) reproduces both numbers, so the labels
# probably round the edges. That is our inference, not something the post states.
