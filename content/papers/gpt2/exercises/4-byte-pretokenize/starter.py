def category(ch):
    """Finished for you: 'space', 'letter', 'digit', or 'other'."""
    if ch.isspace():
        return "space"
    if ch.isalpha():
        return "letter"
    if ch.isdigit():
        return "digit"
    return "other"


def pretokenize(text):
    """Split text into pieces that never merge across categories; a space joins the next run."""
    raise NotImplementedError


def to_bytes(text):
    """UTF-8 bytes of text as a list of ints in 0..255."""
    raise NotImplementedError
