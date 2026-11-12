import numpy as np


def conformer_block(x, ffn1, mhsa, conv, ffn2, final_ln):
    """One Conformer block (Equation 1).

    x: (T, d) array. ffn1, mhsa, conv, ffn2, final_ln: functions that map a
    (T, d) array to a (T, d) array. Return y.
    """
    raise NotImplementedError
