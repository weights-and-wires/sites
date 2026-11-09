def conv_params(k, c_in, c_out):
    """Weights in a k x k convolution from c_in to c_out channels. No bias, no BN."""
    raise NotImplementedError


def basic_block_params(c):
    """Two 3x3 convolutions of width c (Figure 5, left)."""
    raise NotImplementedError


def bottleneck_params(c_out, c_mid):
    """1x1 reduce to c_mid, 3x3 at c_mid, 1x1 restore to c_out (Figure 5, right)."""
    raise NotImplementedError


def projection_shortcut_extra_params(c_out):
    """Extra weights when the identity shortcut of a bottleneck block is a
    1x1 projection from c_out to c_out channels (Equation 2)."""
    raise NotImplementedError
