def timestamp_token(seconds):
    """Return the timestamp token for a time in seconds, on a 20 ms grid, such as "<|2.40|>"."""
    k = int(round(seconds / 0.02))
    return f"<|{k * 0.02:.2f}|>"


def build_target(segments, timestamps):
    """Return the target tokens for a list of (start, end, tokens) segments."""
    out = []
    for start, end, toks in segments:
        if timestamps:
            out += [timestamp_token(start)] + list(toks) + [timestamp_token(end)]
        else:
            out += list(toks)
    return out + ["<|endoftranscript|>"]
