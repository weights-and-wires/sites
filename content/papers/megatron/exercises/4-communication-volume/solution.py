def allreduce_elements_per_layer(b, s, h):
    """Elements all-reduced per transformer layer, forward plus backward."""
    return 4 * b * s * h


def ring_traffic_per_gpu(elements, t):
    """Elements each GPU sends in a ring all-reduce of `elements` elements over t GPUs."""
    return 2 * (t - 1) * elements / t


def logits_elements(b, s, v, fused):
    """Elements communicated for the output layer: b*s*v (all-gather) or b*s (fused loss)."""
    return b * s if fused else b * s * v


def padded_vocab(v, t):
    """Pad v up to a multiple of 128 * t."""
    m = 128 * t
    return -(-v // m) * m
