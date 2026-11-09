def place(kernels, peak, bw):
    """kernels: list of (name, flops, bytes). peak in FLOP/s, bw in bytes/s.

    Return a list of dicts with keys name, intensity, regime, attainable, time.
    """
    raise NotImplementedError
