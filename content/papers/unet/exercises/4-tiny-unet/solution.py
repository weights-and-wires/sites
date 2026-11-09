import numpy as np


def conv2d_valid(x, W, b):
    """Finished for you. x: (Cin, H, W). W: (Cout, Cin, k, k). b: (Cout,). No padding."""
    _, H, Wd = x.shape
    k = W.shape[-1]
    oh, ow = H - k + 1, Wd - k + 1
    out = np.zeros((W.shape[0], oh, ow))
    for i in range(k):
        for j in range(k):
            out += np.einsum("oc,chw->ohw", W[:, :, i, j], x[:, i : i + oh, j : j + ow])
    return out + b[:, None, None]


def relu(x):
    return np.maximum(x, 0.0)


def maxpool2(x):
    """Finished for you. 2x2 max pooling with stride 2 on (C, H, W); H and W must be even."""
    C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, f"pooling needs an even size, got {H} x {W}"
    return x.reshape(C, H // 2, 2, W // 2, 2).max(axis=(2, 4))


def crop_and_concat(skip, up):
    """Finished for you. Center-crop skip to up's size, then concatenate on the channel axis."""
    _, H, W = skip.shape
    _, h, w = up.shape
    top, left = (H - h) // 2, (W - w) // 2
    return np.concatenate([skip[:, top : top + h, left : left + w], up], axis=0)


def upconv2x2(x, W, b):
    """2x2 up-convolution with stride 2.

    x: (Cin, h, w). W: (Cin, Cout, 2, 2). b: (Cout,). Returns (Cout, 2h, 2w), where
    out[o, 2i + a, 2j + c] = sum over c' of x[c', i, j] * W[c', o, a, c] + b[o].
    """
    Cin, h, w = x.shape
    out = np.zeros((W.shape[1], 2 * h, 2 * w))
    for a in range(2):
        for c in range(2):
            out[:, a::2, c::2] = np.einsum("ihw,io->ohw", x, W[:, :, a, c])
    return out + b[:, None, None]


def tiny_unet(x, P):
    """One-level U-Net. x: (1, H, W). P maps a layer name to a (weights, bias) pair."""
    d1 = relu(conv2d_valid(x, *P["c1"]))
    d2 = relu(conv2d_valid(d1, *P["c2"]))  # the skip connection copies this map
    p = maxpool2(d2)
    b1 = relu(conv2d_valid(p, *P["c3"]))
    b2 = relu(conv2d_valid(b1, *P["c4"]))
    u = upconv2x2(b2, *P["up"])
    m = crop_and_concat(d2, u)
    e1 = relu(conv2d_valid(m, *P["c5"]))
    e2 = relu(conv2d_valid(e1, *P["c6"]))
    return conv2d_valid(e2, *P["head"])
