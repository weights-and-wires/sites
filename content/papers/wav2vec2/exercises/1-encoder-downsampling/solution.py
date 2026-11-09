def conv_output_length(n, kernels, strides):
    """Return the number of frames after unpadded convolutions with these kernels and strides.

    n: number of input samples.
    kernels, strides: sequences of equal length, one entry per layer.
    One layer maps n to (n - k) // s + 1, or to 0 when n < k.
    """
    for k, s in zip(kernels, strides):
        if n < k:
            return 0
        n = (n - k) // s + 1
    return n


def receptive_field(kernels, strides):
    """Return (rf, stride): the receptive field in input samples and the total stride."""
    rf, jump = 1, 1
    for k, s in zip(kernels, strides):
        rf += (k - 1) * jump
        jump *= s
    return rf, jump
