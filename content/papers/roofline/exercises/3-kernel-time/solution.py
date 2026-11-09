def place(kernels, peak, bw):
    """kernels: list of (name, flops, bytes). peak in FLOP/s, bw in bytes/s.

    Return a list of dicts with keys name, intensity, regime, attainable, time.
    """
    out = []
    for name, flops, nbytes in kernels:
        intensity = flops / nbytes
        t_compute = flops / peak
        t_memory = nbytes / bw
        regime = "memory" if t_memory > t_compute else "compute"
        out.append({
            "name": name,
            "intensity": intensity,
            "regime": regime,
            "attainable": min(peak, bw * intensity),
            "time": max(t_compute, t_memory),
        })
    return out
