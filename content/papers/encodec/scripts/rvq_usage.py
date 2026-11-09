"""Train a 4-stage RVQ with EMA codebooks and measure codebook usage (Section 3.2).

Run: python3 scripts/rvq_usage.py
"""

import numpy as np

D, K, STAGES, BATCH, STEPS = 8, 64, 4, 512, 300
DECAY = 0.99


def make_data(rng, n):
    centers = np.random.default_rng(123).normal(0, 3.0, size=(10, D))
    return centers[rng.integers(0, 10, n)] + rng.normal(0, 0.5, size=(n, D))


def nearest(cb, x):
    d = (x**2).sum(1, keepdims=True) - 2 * x @ cb.T + (cb**2).sum(1)[None]
    return d.argmin(1)


def run(replace, seed=0):
    rng = np.random.default_rng(seed)
    cbs = [rng.normal(0, 10.0, size=(K, D)) for _ in range(STAGES)]  # badly scaled init
    cnt = [np.ones(K) for _ in range(STAGES)]
    sm = [c.copy() for c in cbs]
    for step in range(STEPS):
        r = make_data(rng, BATCH)
        for s in range(STAGES):
            idx = nearest(cbs[s], r)
            counts = np.bincount(idx, minlength=K).astype(float)
            sums = np.zeros((K, D))
            np.add.at(sums, idx, r)
            cnt[s] = DECAY * cnt[s] + (1 - DECAY) * counts
            sm[s] = DECAY * sm[s] + (1 - DECAY) * sums
            cbs[s] = sm[s] / np.maximum(cnt[s], 1e-5)[:, None]
            if replace:
                dead = cnt[s] < 0.5
                for k in np.flatnonzero(dead):
                    v = r[rng.integers(BATCH)]
                    cbs[s][k], sm[s][k], cnt[s][k] = v, v, 1.0
            r = r - cbs[s][idx]
    # Evaluate on fresh data.
    r = make_data(np.random.default_rng(999), 4096)
    mse0 = (r**2).sum(1).mean()
    out = []
    for s in range(STAGES):
        idx = nearest(cbs[s], r)
        p = np.bincount(idx, minlength=K) / len(idx)
        used = int((p > 0).sum())
        perp = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
        r = r - cbs[s][idx]
        out.append((used, perp, (r**2).sum(1).mean()))
    return mse0, out


for replace in (False, True):
    mse0, out = run(replace)
    print(f"\ndead-code replacement = {replace}; error before quantizing = {mse0:.3f}")
    for s, (used, perp, mse) in enumerate(out, 1):
        print(f"  stage {s}: {used:2d} of {K} codes used, perplexity {perp:5.1f}, error after stage {mse:.3f}")
