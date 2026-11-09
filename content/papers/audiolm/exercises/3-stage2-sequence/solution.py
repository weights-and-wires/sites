import numpy as np


def build_stage2(z, Y, Qp, K, N):
    """Return (seq, loss_mask) for the coarse acoustic stage.

    seq: z, then Y[:, :Qp] flattened row-major with id K + q * N + symbol.
    loss_mask: bool array, True where the model must predict the token.
    """
    z = np.asarray(z)
    coarse = np.asarray(Y)[:, :Qp]
    acoustic = (coarse + np.arange(Qp) * N + K).reshape(-1)
    seq = np.concatenate([z, acoustic])
    mask = np.concatenate([np.zeros(len(z), dtype=bool), np.ones(len(acoustic), dtype=bool)])
    return seq, mask
