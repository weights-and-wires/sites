"""Numbers for the byte-level BPE example (Section 2.2).

The paper says BPE must not merge across character categories, with an
exception for spaces. It doesn't give the rule. The pre-tokenization below
follows the regular expression in OpenAI's released encoder, simplified to
str.isalpha / str.isdigit.

Run: python3 scripts/byte_bpe.py
"""

def category(ch):
    if ch.isspace():
        return "space"
    if ch.isalpha():
        return "letter"
    if ch.isdigit():
        return "digit"
    return "other"


def run_end(text, i):
    """End index of the run of same-category characters that starts at i."""
    cat = category(text[i])
    while i < len(text) and category(text[i]) == cat:
        i += 1
    return i


def pretokenize(text):
    pieces, i, n = [], 0, len(text)
    while i < n:
        if not text[i].isspace():
            j = run_end(text, i)
        else:
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j < n and text[j - 1] == " ":
                # A space joins the run that follows it; earlier whitespace stands alone.
                if j - 1 > i:
                    pieces.append(text[i:j - 1])
                i = j - 1
                j = run_end(text, j)
        pieces.append(text[i:j])
        i = j
    return pieces


print("dog variants:", pretokenize("dog dog. dog! dog?"))
print("pieces:", pretokenize("I have 42 dogs, ok?"))
print("double space:", pretokenize("a  b"))
for s in ["dog", "café", "世界", "\U0001F600"]:
    b = s.encode("utf-8")
    print(f"{s!r}: {len(s)} code point(s), {len(b)} byte(s) = {list(b)}")
print("base vocabulary: 256 byte values; the Unicode code space has 1,114,112 code points")
assert all("".join(pretokenize(t)) == t for t in ["a  b", " x", "x ", "hi\n there", "  "])
