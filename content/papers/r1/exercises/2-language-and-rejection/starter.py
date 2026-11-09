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
    raise NotImplementedError


def keep_response(text, gold, max_paragraph_words=60):
    """True when the response is well formed, correct, single-language, and has no code fence or long paragraph."""
    raise NotImplementedError


def rejection_sample(responses, gold, k):
    """Return the first k responses that keep_response accepts, in order."""
    raise NotImplementedError
