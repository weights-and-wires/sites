def timestamp_token(seconds):
    """Return the timestamp token for a time in seconds, on a 20 ms grid, such as "<|2.40|>"."""
    raise NotImplementedError


def build_target(segments, timestamps):
    """Return the target tokens for a list of (start, end, tokens) segments."""
    raise NotImplementedError
