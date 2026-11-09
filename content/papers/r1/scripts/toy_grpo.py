"""A toy GRPO loop on two-digit addition with the Equation 4 rewards.

The policy is a softmax over six behaviors. Five write tags and differ in the
answer they give; one skips the tags. There is no KL term, and the clip does
nothing because every batch is used for one update.
"""

import numpy as np

from r1_lib import group_advantages, rule_reward

BEHAVIORS = [(True, 0), (True, 1), (True, -1), (True, 10), (True, -10), (False, 0)]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def generate(k, a, b):
    tagged, delta = BEHAVIORS[k]
    ans = a + b + delta
    if tagged:
        return f"<think>{a} + {b}</think><answer>{ans}</answer>"
    return f"The answer is {ans}."


def grpo_step(logits, behaviors, rewards, lr):
    adv = group_advantages(rewards)
    p = softmax(logits)
    grad = np.zeros_like(logits)
    for k, A in zip(behaviors, adv):
        g = -p.copy()
        g[k] += 1.0
        grad += A * g
    return logits + lr * grad / len(adv)


def train(seed=0, steps=60, G=16, lr=1.0):
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


if __name__ == "__main__":
    logits, history = train()
    print("P(tagged and correct) at step 0, 10, 20, 40, 60:", [round(float(history[i]), 4) for i in (0, 10, 20, 40, 60)])
    print("final probabilities:", np.round(softmax(logits), 4).tolist())

    # The one-step example used by the exercise tests.
    z = np.zeros(6)
    behaviors = [0, 0, 1, 5]
    rewards = [2, 2, 1, 0]
    print("advantages:", np.round(group_advantages(rewards), 4).tolist())
    print("one step, lr = 1:", np.round(grpo_step(z, behaviors, rewards, 1.0), 4).tolist())
