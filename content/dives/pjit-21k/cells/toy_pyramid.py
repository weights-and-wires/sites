# Part 2: a toy pyramid-readout experiment. Our own toy numbers, not the post's.
# A 1-D "image" of 32 values: three smooth waves plus a small fast wave.
# Trunk: 6 residual tanh blocks, width 64, predicts x from [z, t]. Hand-written backprop.
# With readouts, a head after block 1 predicts the 4x-pooled x and a head after
# block 3 predicts the 2x-pooled x. All losses use the weight 1 / max(t, 0.1)^2.
import numpy as np

D, L, W = 32, 6, 64

def make(r, b):
    u = np.arange(D) / D
    a = r.normal(size=(b, 3)); ph = r.uniform(0, 6.28, size=(b, 3))
    s = sum(a[:, [k]] * np.sin(2 * np.pi * (k + 1) * u + ph[:, [k]]) for k in range(3))
    h = 0.3 * r.normal(size=(b, 1)) * np.sin(2 * np.pi * 12 * u + r.uniform(0, 6.28, size=(b, 1)))
    return s + h

def down(x, f): return x.reshape(x.shape[0], -1, f).mean(-1)

def init(r):
    P = {"W0": r.normal(0, .15, (D + 1, W))}
    for i in range(1, L): P[f"B{i}"] = r.normal(0, .15, (W, W))
    for name, n in (("H4", D // 4), ("H2", D // 2), ("H1", D)): P[name] = r.normal(0, .1, (W, n))
    return P

def forward(P, inp):
    hs = [np.tanh(inp @ P["W0"])]
    for i in range(1, L): hs.append(hs[-1] + np.tanh(hs[-1] @ P[f"B{i}"]))
    return hs

def grads(P, x, t, z, readouts):
    inp = np.concatenate([z, t], 1)
    hs = forward(P, inp)
    w = 1 / np.maximum(t, 0.1) ** 2
    G = {k: np.zeros_like(v) for k, v in P.items()}
    dh = [np.zeros_like(h) for h in hs]
    heads = [(L - 1, "H1", x)]
    if readouts: heads += [(1, "H4", down(x, 4)), (3, "H2", down(x, 2))]
    for li, name, tgt in heads:
        pred = hs[li] @ P[name]
        d = 2 * w * (pred - tgt) / pred.size
        G[name] += hs[li].T @ d
        dh[li] += d @ P[name].T
    for i in range(L - 1, 0, -1):
        g = dh[i] * (1 - np.tanh(hs[i - 1] @ P[f"B{i}"]) ** 2)
        G[f"B{i}"] += hs[i - 1].T @ g
        dh[i - 1] += dh[i] + g @ P[f"B{i}"].T
    G["W0"] += inp.T @ (dh[0] * (1 - hs[0] ** 2))
    return G

def noisy(r, b, t0=None):
    x = make(r, b)
    t = np.full((b, 1), t0) if t0 is not None else 1 / (1 + np.exp(-r.normal(-0.2, 1.0, (b, 1))))
    return x, t, (1 - t) * x + t * r.normal(size=x.shape)

def train(readouts, seed, steps=200, lr=3e-3):
    r = np.random.default_rng(seed)
    P = init(r)
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    for it in range(1, steps + 1):
        x, t, z = noisy(r, 64)
        G = grads(P, x, t, z, readouts)
        for k in P:
            m[k] = .9 * m[k] + .1 * G[k]; v2[k] = .999 * v2[k] + .001 * G[k] ** 2
            P[k] -= lr * (m[k] / (1 - .9 ** it)) / (np.sqrt(v2[k] / (1 - .999 ** it)) + 1e-8)
    r2 = np.random.default_rng(99)
    out = []
    for t0 in (0.3, 0.6, 0.9):
        x, t, z = noisy(r2, 512, t0)
        out.append(float(np.mean((forward(P, np.concatenate([z, t], 1))[-1] @ P["H1"] - x) ** 2)))
    return out

# Gradient reaching the first block at initialization, with and without readouts.
r = np.random.default_rng(5)
P0 = init(r)
norms = {False: [], True: []}
for _ in range(10):
    x, t, z = noisy(r, 64)
    for ro in (False, True):
        norms[ro].append(np.linalg.norm(grads(P0, x, t, z, ro)["W0"]))
g1, g3 = np.mean(norms[False]), np.mean(norms[True])
print(f"mean gradient norm on block 0 at init: final head only {g1:.4f}, with readouts {g3:.4f} ({g3 / g1:.2f}x)")

seeds = (0, 1, 2)
runs = {ro: np.array([train(ro, s) for s in seeds]) for ro in (False, True)}
print("\nfinal-head x-prediction MSE, mean +/- sd over", len(seeds), "seeds (200 steps each)")
print("  t:            0.3             0.6             0.9")
for ro, name in ((False, "final head only "), (True, "3 readouts      ")):
    m, s = runs[ro].mean(0), runs[ro].std(0)
    print(f"  {name}" + "  ".join(f"{a:.3f} +/- {b:.3f}" for a, b in zip(m, s)))
diff = runs[True].mean(0) - runs[False].mean(0)
spread = np.maximum(runs[True].std(0), runs[False].std(0))
print("  readouts minus final-only: " + "  ".join(f"{d:+.3f}" for d in diff))
print("  that difference in seed sds: " + "  ".join(f"{d / sp:+.1f}" for d, sp in zip(diff, spread)))
if np.all(np.abs(diff) <= 1.5 * spread):
    print("Every gap is within about one sd: this toy does not show a benefit from the readouts.")
else:
    print("Some gap exceeds 1.5 sd. Check it with more seeds before you believe it.")
result = {
    "t": [0.3, 0.6, 0.9],
    "one": {"mean": runs[False].mean(0), "sd": runs[False].std(0)},
    "three": {"mean": runs[True].mean(0), "sd": runs[True].std(0)},
    "grad": {"one": float(g1), "three": float(g3)},
}
