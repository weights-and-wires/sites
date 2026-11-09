import re

THINK_ANSWER = re.compile(r"<think>(.+?)</think>\s*<answer>(.+?)</answer>", re.DOTALL)


def split_response(text):
    """Finished for you. Return (think, answer) for a well-formed response, else None."""
    m = THINK_ANSWER.fullmatch(text.strip())
    if m is None or not m.group(1).strip() or not m.group(2).strip():
        return None
    return m.group(1).strip(), m.group(2).strip()


def language_reward(cot):
    """Equation 7 with this page's word rule: share of ASCII-letter words among words with letters."""
    words = [w for w in cot.split() if any(c.isalpha() for c in w)]
    if not words:
        return 0.0
    target = [w for w in words if all(c.isascii() for c in w if c.isalpha())]
    return len(target) / len(words)


def keep_response(text, gold, max_paragraph_words=60):
    """True when the response is well formed, correct, single-language, and has no code fence or long paragraph."""
    parts = split_response(text)
    if parts is None:
        return False
    think, answer = parts
    if not re.fullmatch(r"-?\d+", answer) or int(answer) != gold:
        return False
    if language_reward(think) != 1.0:
        return False
    if "```" in think:
        return False
    if any(len(p.split()) > max_paragraph_words for p in think.split("\n\n")):
        return False
    return True


def rejection_sample(responses, gold, k):
    """Return the first k responses that keep_response accepts, in order."""
    kept = []
    for r in responses:
        if len(kept) == k:
            break
        if keep_response(r, gold):
            kept.append(r)
    return kept
