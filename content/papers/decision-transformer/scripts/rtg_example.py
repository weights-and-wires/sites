#!/usr/bin/env python3
"""Returns-to-go (Section 3, Equation 2) and the target-return update.

The paper indexes timesteps from 1; NumPy indexes from 0. The code below keeps
the paper's formula, R_t = sum of r_t' for t' >= t, and shifts the index.
"""

import numpy as np


def returns_to_go(rewards):
    """R_t = sum_{t' >= t} r_t' with no discount."""
    r = np.asarray(rewards, dtype=float)
    return np.cumsum(r[::-1])[::-1]


def main():
    # A sparse episode: the only reward comes at the last step.
    sparse = [0, 0, 0, 1]
    print("sparse rewards      :", sparse)
    print("sparse returns-to-go:", returns_to_go(sparse).tolist())

    # A mixed episode.
    mixed = [1, 0, 2, -1]
    rtg = returns_to_go(mixed)
    print("mixed rewards       :", mixed)
    print("mixed returns-to-go :", rtg.tolist())

    # The same numbers counted from the start of the episode (the wrong signal).
    print("reward so far       :", np.cumsum(mixed).tolist())

    # The test-time update: R_{t+1} = R_t - r_t starting from a target.
    target = 2.0
    R = [target]
    for r in mixed[:-1]:
        R.append(R[-1] - r)
    print("target return 2, decremented by the achieved rewards:", R)
    print("matches the dataset's returns-to-go:", np.allclose(R, rtg))

    # The sliding context of the evaluation loop: K = 3, reward +1 at each of 6 steps.
    K, R, t = 3, [6.0], [1]
    print("evaluation loop, K = 3, reward +1 per step, target 6:")
    for step in range(1, 7):
        n = min(K, len(R))
        print(f"  step {step}: returns-to-go {R[-n:]}, timesteps {t[-n:]}")
        R.append(R[-1] - 1.0)
        t.append(t[-1] + 1)

    # Two episodes with the same total reward but a different reward timing.
    early = [2, 0, 0, 0]
    late = [0, 0, 0, 2]
    print("early episode returns-to-go:", returns_to_go(early).tolist())
    print("late episode returns-to-go :", returns_to_go(late).tolist())


if __name__ == "__main__":
    main()
