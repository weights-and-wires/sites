"""Build the text stream W from timed words (Section 3.4.4, Equation 5).

Run: python3 scripts/text_alignment.py
"""

PAD, EPAD = "PAD", "EPAD"


def align(words, T):
    """words: list of (start index t_i, tokens), 1-based starts. Returns W with T slots."""
    W = [PAD] * T
    for t, toks in words:
        if t == 1:
            W[0] = EPAD
            t = 2
        elif W[t - 2] == PAD:  # slot t-1, 0-based t-2; never overwrite a text token
            W[t - 2] = EPAD
        for j, tok in enumerate(toks):
            if t - 1 + j < T:
                W[t - 1 + j] = tok
    return W


words = [(1, ["Hello", ","]), (5, ["I", "'", "m"]), (9, ["Moshi"])]
W = align(words, 12)
print(W)
print("PAD count    =", W.count(PAD), "of", len(W))
print("PAD fraction =", round(W.count(PAD) / len(W), 4))
# Adjacent words: the EPAD would overwrite a text token, so it is skipped.
print(align([(2, ["a", "b"]), (4, ["c"])], 6))
# At 65 percent PAD (Section 3.4.4) and 12.5 slots per second:
print("non-PAD slots per second at 65% PAD =", round(12.5 * 0.35, 3))
