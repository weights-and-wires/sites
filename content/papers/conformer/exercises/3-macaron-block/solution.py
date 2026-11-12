import numpy as np


def conformer_block(x, ffn1, mhsa, conv, ffn2, final_ln):
    """One Conformer block (Equation 1).

    x: (T, d) array. ffn1, mhsa, conv, ffn2, final_ln: functions that map a
    (T, d) array to a (T, d) array. Return y.
    """
    x1 = x + 0.5 * ffn1(x)
    x2 = x1 + mhsa(x1)
    x3 = x2 + conv(x2)
    return final_ln(x3 + 0.5 * ffn2(x3))
