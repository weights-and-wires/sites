import re

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_ANSWER = re.compile(r"<think>(.+?)</think>\s*<answer>(.+?)</answer>", re.DOTALL)
INTEGER = re.compile(r"-?\d+")


def extract_answer(text):
    """Return the stripped content of the last <answer>...</answer> pair, or None."""
    found = ANSWER_TAG.findall(text)
    return found[-1].strip() if found else None


def accuracy_reward(text, gold):
    """Return 1.0 if the extracted answer is an integer equal to gold, else 0.0."""
    ans = extract_answer(text)
    if ans is None or not INTEGER.fullmatch(ans):
        return 0.0
    return 1.0 if int(ans) == gold else 0.0


def format_reward(text):
    """Return 1.0 for exactly one non-blank <think> block then one non-blank <answer> block."""
    body = text.strip()
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if body.count(tag) != 1:
            return 0.0
    m = THINK_ANSWER.fullmatch(body)
    if m is None or not m.group(1).strip() or not m.group(2).strip():
        return 0.0
    return 1.0


def rule_reward(text, gold):
    """Equation 4: accuracy reward plus format reward, equal weights."""
    return accuracy_reward(text, gold) + format_reward(text)
