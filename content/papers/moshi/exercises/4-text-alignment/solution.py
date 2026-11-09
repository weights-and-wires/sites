PAD, EPAD = "PAD", "EPAD"


def align_text(words, T):
    """Return the T-slot text stream for words given as (start step, tokens)."""
    W = [PAD] * T
    for t, tokens in words:
        if t == 1:
            W[0] = EPAD
            t = 2
        elif W[t - 2] == PAD:
            W[t - 2] = EPAD
        for j, tok in enumerate(tokens):
            if t - 1 + j < T:
                W[t - 1 + j] = tok
    return W
