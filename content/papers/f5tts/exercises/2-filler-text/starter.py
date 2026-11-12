def pad_text(chars, n_frames, filler="<F>"):
    """Return a new list: chars, then n_frames - len(chars) fillers.

    Raise ValueError if len(chars) > n_frames. Do not modify chars.
    """
    raise NotImplementedError


def estimate_total_frames(ref_frames, ref_text, gen_text):
    """Return the total mel length N as an int.

    Generated frames = floor(ref_frames * len(gen_text) / len(ref_text)).
    Total = ref_frames + generated frames.
    """
    raise NotImplementedError
