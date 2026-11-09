def category(ch):
    """Finished for you: 'space', 'letter', 'digit', or 'other'."""
    if ch.isspace():
        return "space"
    if ch.isalpha():
        return "letter"
    if ch.isdigit():
        return "digit"
    return "other"


def _run_end(text, i):
    cat = category(text[i])
    while i < len(text) and category(text[i]) == cat:
        i += 1
    return i


def pretokenize(text):
    """Split text into pieces that never merge across categories; a space joins the next run."""
    pieces, i, n = [], 0, len(text)
    while i < n:
        if not text[i].isspace():
            j = _run_end(text, i)
        else:
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j - 1] == " ":
                if j - 1 > i:
                    pieces.append(text[i:j - 1])
                i = j - 1
                j = _run_end(text, j)
        pieces.append(text[i:j])
        i = j
    return pieces


def to_bytes(text):
    """UTF-8 bytes of text as a list of ints in 0..255."""
    return list(text.encode("utf-8"))
