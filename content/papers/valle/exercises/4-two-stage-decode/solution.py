import numpy as np


def generate(ar_step, nar_step, phonemes, prompt_codes, max_len, eos, K=8):
    """Return the generated code matrix of shape (T, K), without the prompt.

    prompt_codes: (T_prompt, K) integer array of the enrolled recording.
    """
    seq = [int(c) for c in prompt_codes[:, 0]]
    first = []
    while len(first) < max_len:
        tok = ar_step(phonemes, list(seq))
        if tok == eos:
            break
        seq.append(tok)
        first.append(tok)
    T = len(first)
    codes = np.zeros((T, K), dtype=int)
    codes[:, 0] = first
    for j in range(2, K + 1):
        codes[:, j - 1] = nar_step(phonemes, prompt_codes, codes[:, : j - 1].copy(), j)
    return codes
