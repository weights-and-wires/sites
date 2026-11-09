"""Shared helpers for the r1 page scripts: rule-based rewards (Equation 4),
the language-consistency reward (Equation 7), and the group advantage
(Equation 3). They match the solutions to the exercises."""

import re

import numpy as np

THINK_ANSWER = re.compile(r"<think>(.+?)</think>\s*<answer>(.+?)</answer>", re.DOTALL)
ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
INTEGER = re.compile(r"-?\d+")


def extract_answer(text):
    found = ANSWER_TAG.findall(text)
    return found[-1].strip() if found else None


def accuracy_reward(text, gold):
    ans = extract_answer(text)
    if ans is None or not INTEGER.fullmatch(ans):
        return 0.0
    return 1.0 if int(ans) == gold else 0.0


def format_reward(text):
    body = text.strip()
    if body.count("<think>") != 1 or body.count("</think>") != 1:
        return 0.0
    if body.count("<answer>") != 1 or body.count("</answer>") != 1:
        return 0.0
    m = THINK_ANSWER.fullmatch(body)
    if m is None or not m.group(1).strip() or not m.group(2).strip():
        return 0.0
    return 1.0


def rule_reward(text, gold):
    return accuracy_reward(text, gold) + format_reward(text)


def language_reward(cot):
    words = [w for w in cot.split() if any(c.isalpha() for c in w)]
    if not words:
        return 0.0
    target = [w for w in words if all(c.isascii() for c in w if c.isalpha())]
    return len(target) / len(words)


def group_advantages(rewards, eps=1e-6):
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)
