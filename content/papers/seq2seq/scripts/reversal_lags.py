"""Minimal time lag with and without source reversal (Section 3.3).

Concatenate a source of n tokens and a target of n tokens whose j-th token
is the translation of the j-th source token. Measure the distance between each
source token and its target token in the concatenated sequence.
"""

import numpy as np


def lags(n, reverse):
    out = []
    for j in range(n):
        src_pos = (n - 1 - j) if reverse else j  # where source token j sits
        tgt_pos = n + j  # the target follows the source
        out.append(tgt_pos - src_pos)
    return np.array(out)


if __name__ == "__main__":
    for n in (4, 5, 25):
        for reverse in (False, True):
            l = lags(n, reverse)
            shown = l.tolist() if n <= 5 else "..."
            print(f"n={n:2d} reverse={str(reverse):5s} lags={shown} mean={l.mean():.1f} min={l.min()} max={l.max()}")
