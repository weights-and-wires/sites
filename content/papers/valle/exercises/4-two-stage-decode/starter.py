import numpy as np


def generate(ar_step, nar_step, phonemes, prompt_codes, max_len, eos, K=8):
    """Return the generated code matrix of shape (T, K), without the prompt.

    prompt_codes: (T_prompt, K) integer array of the enrolled recording.
    """
    raise NotImplementedError
