import numpy as np


def score(option, mode):
    """Score one completion. option: {"ctx": [...], "unc": [...] or None} of per-token log-probs."""
    ctx = np.asarray(option["ctx"], dtype=float)
    if mode == "sum":
        return float(ctx.sum())
    if mode == "mean":
        return float(ctx.mean())
    if mode == "ratio":
        if option.get("unc") is None:
            raise ValueError("ratio scoring needs the answer-context log-probabilities")
        unc = np.asarray(option["unc"], dtype=float)
        return float(ctx.sum() - unc.sum())
    raise ValueError(f"unknown mode {mode!r}")


def choose(options, mode):
    """Index of the best option under the given scoring mode."""
    return int(np.argmax([score(o, mode) for o in options]))
