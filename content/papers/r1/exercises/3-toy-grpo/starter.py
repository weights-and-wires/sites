import re

import numpy as np

# (writes tags, offset added to the true sum)
BEHAVIORS = [(True, 0), (True, 1), (True, -1), (True, 10), (True, -10), (False, 0)]
ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_ANSWER = re.compile(r"<think>(.+?)</think>\s*<answer>(.+?)</answer>", re.DOTALL)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def generate(k, a, b):
    """Finished for you. The text that behavior k writes for the question a + b."""
    tagged, delta = BEHAVIORS[k]
    ans = a + b + delta
    if tagged:
        return f"<think>{a} + {b}</think><answer>{ans}</answer>"
    return f"The answer is {ans}."


def rule_reward(text, gold):
    """Finished for you. Equation 4: accuracy plus format, each 0 or 1."""
    found = ANSWER_TAG.findall(text)
    acc = 1.0 if found and found[-1].strip() == str(gold) else 0.0
    fmt = 1.0 if THINK_ANSWER.fullmatch(text.strip()) else 0.0
    return acc + fmt


def grpo_step(logits, behaviors, rewards, lr):
    """One group-relative policy-gradient step on a softmax policy. Return new logits."""
    raise NotImplementedError


def train(seed=0, steps=60, G=16, lr=1.0):
    """Finished for you. Returns the final logits and P(behavior 0) after each step."""
    rng = np.random.default_rng(seed)
    logits = np.zeros(len(BEHAVIORS))
    history = [softmax(logits)[0]]
    for _ in range(steps):
        a, b = rng.integers(10, 100, size=2)
        ks = rng.choice(len(BEHAVIORS), size=G, p=softmax(logits))
        rewards = [rule_reward(generate(k, a, b), a + b) for k in ks]
        logits = grpo_step(logits, ks, rewards, lr)
        history.append(softmax(logits)[0])
    return logits, history
