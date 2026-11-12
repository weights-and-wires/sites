# Part 1: shapes, the x-to-v loss, and the timestep schedule.
# Everything here is arithmetic from the post's settings. No training happens.
import math
import numpy as np

# 1. Patch tokens and readout shapes at the 512 stage.
side, patch = 512, 32
g = side // patch                       # 16 tokens per side
tokens = g * g
print("512 x 512, 32 x 32 patches ->", tokens, "image tokens; each token holds", patch * patch * 3, "values")

def unpatchify(tok, grid, p):
    """(grid*grid, p*p*3) token outputs -> (grid*p, grid*p, 3) image."""
    img = tok.reshape(grid, grid, p, p, 3)
    return img.transpose(0, 2, 1, 3, 4).reshape(grid * p, grid * p, 3)

for name, block, p in (("readout 1", 10, 8), ("readout 2", 16, 16), ("final head", 22, 32)):
    head_out = np.zeros((tokens, p * p * 3))        # one linear head per readout
    img = unpatchify(head_out, g, p)
    print(f"  after block {block:2d}: head emits {p}*{p}*3 = {p*p*3:4d} values per token -> image {img.shape[:2]}")

# 2. Why x-prediction with a v-loss weights low-noise errors by 1 / max(t, 0.1)^2.
#    Convention (the post's plot runs clean to noise): z = (1 - t) x + t * sigma * eps, so t = 1 is pure noise.
rng = np.random.default_rng(0)
x = rng.normal(size=8)
eps = rng.normal(size=8)
sigma, t = 2.0, 0.3
z = (1 - t) * x + t * sigma * eps
v = sigma * eps - x                     # true velocity dz/dt
x_hat = x + 0.1 * rng.normal(size=8)    # an imperfect x-prediction
v_hat = (z - x_hat) / t                 # the velocity the x-prediction implies
print("\nv_hat - v equals (x - x_hat) / t:", np.allclose(v_hat - v, (x - x_hat) / t))
print("loss weight 1 / max(t, 0.1)^2 at different t:")
for tt in (0.05, 0.1, 0.3, 0.6, 0.9):
    print(f"  t = {tt:.2f}  weight = {1 / max(tt, 0.1) ** 2:6.2f}")

# 3. Share of timesteps in the band around t = 0.10, for both schedules.
def Phi(u): return 0.5 * (1 + math.erf(u / math.sqrt(2)))
def logit(p): return math.log(p / (1 - p))
def exact(mu, s, a, b): return Phi((logit(b) - mu) / s) - Phi((logit(a) - mu) / s)

band = (0.075, 0.125)
print("\nshare of timesteps with t in", band)
shares = {}
for label, (mu, s) in (("phase 1 LogitNormal(0.8, 0.8)", (0.8, 0.8)), ("phase 2 LogitNormal(-0.2, 1.0)", (-0.2, 1.0))):
    t_samples = 1 / (1 + np.exp(-rng.normal(mu, s, 2_000_000)))
    mc = np.mean((t_samples > band[0]) & (t_samples < band[1]))
    shares[label] = exact(mu, s, *band)
    print(f"  {label}: exact {100 * shares[label]:.4f}%, Monte Carlo {100 * mc:.4f}%, mean t {t_samples.mean():.3f}")
ratio = list(shares.values())[1] / list(shares.values())[0]
print(f"  ratio phase 2 / phase 1: {ratio:.0f}x")
result = {"tokens": tokens, "ratio": ratio}
