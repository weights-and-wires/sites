import numpy as np


def output_lattice(f, g, y):
    """Return (emit, null) for the RNN transducer lattice.

    f: (T, K+1) transcription vectors, column 0 is the null symbol.
    g: (U+1, K+1) prediction vectors.
    y: tuple of U labels in 1..K.
    emit: (T, U), emit[t, u] = Pr(y[u] | t, u).
    null: (T, U+1), null[t, u] = Pr(null | t, u).
    """
    raise NotImplementedError
