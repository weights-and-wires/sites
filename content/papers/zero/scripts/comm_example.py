"""Communication volume of data parallelism and ZeRO-DP, in units of Psi.

Section 7: a reduce-scatter or an all-gather moves Psi elements per process, and
an all-reduce is one of each, so standard data parallelism moves 2 Psi. This
script also checks with NumPy that reduce-scatter followed by all-gather gives
the same result as an all-reduce.
"""

import numpy as np


def volume(stage):
    if stage == 0:
        return 2  # all-reduce of gradients
    if stage in (1, 2):
        return 1 + 1  # reduce-scatter of gradients + all-gather of updated parameters
    return 1 + 1 + 1  # reduce-scatter + forward all-gather + backward all-gather


def main():
    for stage in range(4):
        v = volume(stage)
        print(f"stage {stage}: {v} Psi, {v / 2:.1f}x the baseline")

    rng = np.random.default_rng(0)
    nd, psi = 4, 8
    grads = rng.integers(-5, 6, size=(nd, psi)).astype(float)
    print("\nWorker gradients:")
    print(grads)
    total = grads.sum(axis=0)
    shards = np.array_split(total, nd)  # reduce-scatter: worker i keeps the sum of chunk i
    print("reduce-scatter, worker 0 keeps:", shards[0], " worker 3 keeps:", shards[3])
    gathered = np.concatenate(shards)  # all-gather
    print("all-gather result:", gathered)
    print("equals all-reduce:", np.array_equal(gathered, total))


if __name__ == "__main__":
    main()
