import numpy as np


def dot(a16, b16, acc_dtype):
    """Sequential dot product with an accumulator of type acc_dtype.

    Returns a np.float16 scalar.
    """
    s = acc_dtype(0)
    for x, y in zip(a16, b16):
        s = acc_dtype(s + acc_dtype(x) * acc_dtype(y))
    return np.float16(s)
