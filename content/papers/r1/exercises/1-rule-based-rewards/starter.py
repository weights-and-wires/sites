import re

ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_ANSWER = re.compile(r"<think>(.+?)</think>\s*<answer>(.+?)</answer>", re.DOTALL)
INTEGER = re.compile(r"-?\d+")


def extract_answer(text):
    """Return the stripped content of the last <answer>...</answer> pair, or None."""
    raise NotImplementedError


def accuracy_reward(text, gold):
    """Return 1.0 if the extracted answer is an integer equal to gold, else 0.0."""
    raise NotImplementedError


def format_reward(text):
    """Return 1.0 for exactly one non-blank <think> block then one non-blank <answer> block."""
    raise NotImplementedError


def rule_reward(text, gold):
    """Equation 4: accuracy reward plus format reward, equal weights."""
    raise NotImplementedError
