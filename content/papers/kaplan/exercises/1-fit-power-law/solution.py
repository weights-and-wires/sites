import numpy as np


def fit_power_law(x, y):
    """Fit y = (x_c / x) ** alpha by least squares on log y against log x.

    Returns (alpha, x_c), with alpha > 0 for a loss that falls as x grows.
    """
    lx, ly = np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    alpha = -slope
    x_c = np.exp(intercept / alpha)
    return float(alpha), float(x_c)


def predict(alpha, x_c, x):
    """Evaluate (x_c / x) ** alpha."""
    return (x_c / np.asarray(x, dtype=float)) ** alpha


def extrapolate(x, y, x_new):
    """Predict y at x_new and give the one-sigma uncertainty of a new observation.

    Returns (y_pred, sd_log). sd_log is the standard deviation of log y for a
    new measurement at x_new, so exp(+-sd_log) scales y_pred up and down.
    """
    lx, ly = np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float))
    n = lx.size
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    s2 = resid @ resid / (n - 2)
    sxx = ((lx - lx.mean()) ** 2).sum()
    lnew = np.log(np.asarray(x_new, dtype=float))
    y_pred = np.exp(slope * lnew + intercept)
    sd_log = np.sqrt(s2 * (1 + 1 / n + (lnew - lx.mean()) ** 2 / sxx))
    return y_pred, sd_log
