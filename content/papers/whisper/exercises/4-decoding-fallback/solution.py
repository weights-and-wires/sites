import zlib


def compression_ratio(text):
    """Return len(utf-8 bytes) / len(zlib-compressed bytes)."""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b))


def needs_fallback(text, avg_logprob, ratio_threshold=2.4, logprob_threshold=-1.0):
    """Return True when the decode looks bad: too repetitive or too unlikely."""
    return compression_ratio(text) > ratio_threshold or avg_logprob < logprob_threshold


def temperature_schedule(start=0.0, step=0.2, stop=1.0):
    """Return the temperatures start, start + step, ..., stop as a list of floats, stop included."""
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


def decode_with_fallback(decode, temperatures):
    """Try each temperature in order; return (text, avg_logprob, temperature) for the first good decode.

    decode(t) returns (text, avg_logprob). If none is good, return the last result.
    """
    result = None
    for t in temperatures:
        text, lp = decode(t)
        result = (text, lp, t)
        if not needs_fallback(text, lp):
            return result
    return result
