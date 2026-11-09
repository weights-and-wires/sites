"""Equation 1 on a five-token target, with teacher-forced decoder inputs."""

import numpy as np

VOCAB = ["W", "X", "Y", "Z", "<EOS>"]
TARGET = [0, 1, 2, 3, 4]  # W X Y Z <EOS>, the target of Figure 1

# Decoder logits, one row per step. Row t is the output after teacher-forced input t.
LOGITS = np.array([
    [2.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 3.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 2.0],
])


def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


if __name__ == "__main__":
    inputs = ["<EOS>"] + [VOCAB[i] for i in TARGET[:-1]]
    print("decoder inputs:", inputs)
    lp = log_softmax(LOGITS)
    step = np.array([lp[t, y] for t, y in enumerate(TARGET)])
    for t, y in enumerate(TARGET):
        print(f"t={t + 1} p({VOCAB[y]} | ...) = {np.exp(step[t]):.4f}  log = {step[t]:.4f}")
    print(f"product of probabilities = {np.exp(step).prod():.6f}")
    print(f"sum of log probabilities = {step.sum():.4f}")
    print(f"exp(sum of logs)         = {np.exp(step.sum()):.6f}")
    print(f"per-token perplexity     = {np.exp(-step.sum() / len(TARGET)):.4f}")
    print(f"mean over 2 pairs (this one and one with log p = -3.0) = {(step.sum() - 3.0) / 2:.4f}")
