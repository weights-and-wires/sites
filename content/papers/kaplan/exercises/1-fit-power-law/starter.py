import numpy as np


def fit_power_law(x, y):
    """Fit y = (x_c / x) ** alpha by least squares on log y against log x.

    Returns (alpha, x_c), with alpha > 0 for a loss that falls as x grows.
    """
    raise NotImplementedError


def predict(alpha, x_c, x):
    """Evaluate (x_c / x) ** alpha."""
    raise NotImplementedError


def extrapolate(x, y, x_new):
    """Predict y at x_new and give the one-sigma uncertainty of a new observation.

    Returns (y_pred, sd_log). sd_log is the standard deviation of log y for a
    new measurement at x_new, so exp(+-sd_log) scales y_pred up and down.
    """
    raise NotImplementedError
