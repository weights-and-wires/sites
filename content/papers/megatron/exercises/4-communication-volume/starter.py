def allreduce_elements_per_layer(b, s, h):
    """Elements all-reduced per transformer layer, forward plus backward."""
    raise NotImplementedError


def ring_traffic_per_gpu(elements, t):
    """Elements each GPU sends in a ring all-reduce of `elements` elements over t GPUs."""
    raise NotImplementedError


def logits_elements(b, s, v, fused):
    """Elements communicated for the output layer: b*s*v (all-gather) or b*s (fused loss)."""
    raise NotImplementedError


def padded_vocab(v, t):
    """Pad v up to a multiple of 128 * t."""
    raise NotImplementedError
