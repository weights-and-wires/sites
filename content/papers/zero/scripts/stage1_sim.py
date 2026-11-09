"""Simulate ZeRO stage 1 (P_os) on 4 workers with Adam, in NumPy.

Each worker holds the full fp32 parameter vector but only 1/Nd of the Adam
state (Section 5.1). After the step every worker holds identical, updated
parameters, and they match a plain data-parallel Adam step on one big state.
"""

import numpy as np

LR, B1, B2, EPS = 0.1, 0.9, 0.999, 1e-8


def adam(p, g, m, v, t):
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    mh = m / (1 - B1**t)
    vh = v / (1 - B2**t)
    return p - LR * mh / (np.sqrt(vh) + EPS), m, v


def main():
    nd, psi = 4, 10
    rng = np.random.default_rng(0)
    p0 = rng.normal(size=psi)
    local_grads = rng.normal(size=(nd, psi))
    g = local_grads.mean(axis=0)  # data parallel: averaged gradient

    # Baseline: every worker keeps full m and v.
    p_ddp, m, v = adam(p0, g, np.zeros(psi), np.zeros(psi), 1)

    # ZeRO-1: worker i owns slice i.
    slices = np.array_split(np.arange(psi), nd)
    m_sh = [np.zeros(len(s)) for s in slices]
    v_sh = [np.zeros(len(s)) for s in slices]
    new = []
    for i, s in enumerate(slices):
        p_i, m_sh[i], v_sh[i] = adam(p0[s], g[s], m_sh[i], v_sh[i], 1)
        new.append(p_i)
    p_zero = np.concatenate(new)  # all-gather

    print("slice sizes:", [len(s) for s in slices])
    print("Adam state floats per worker, DDP:", 2 * psi, " ZeRO-1:", [2 * len(s) for s in slices])
    print("parameters match DDP:", np.allclose(p_zero, p_ddp))
    print("max abs difference:", float(np.abs(p_zero - p_ddp).max()))
    print("first update, DDP:   ", np.round(p_ddp[:3], 4))
    print("first update, ZeRO-1:", np.round(p_zero[:3], 4))

    # Bytes: 12 bytes of optimizer state per parameter (fp32 copy + m + v).
    print("optimizer bytes per worker, DDP:", 12 * psi, " ZeRO-1:", [12 * len(s) for s in slices])


if __name__ == "__main__":
    main()
