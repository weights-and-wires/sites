import numpy as np


def welch_t(a, b):
    """Welch's t statistic and Welch-Satterthwaite degrees of freedom for two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    va = a.var(ddof=1) / len(a)
    vb = b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return float(t), float(df)
