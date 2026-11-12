import numpy as np


def expected_length(parents, accept):
    """Expected tokens per round: 1 + sum of P(node accepted).

    parents: parent index of each node, -1 for a child of the last context token.
    accept:  P(node accepted | it is tested) for each node.
    """
    n = len(parents)
    reach = np.zeros(n)
    remaining = {}
    total = 0.0
    for i in range(n):
        p = parents[i]
        if p not in remaining:
            remaining[p] = 1.0 if p < 0 else reach[p]
        reach[i] = remaining[p] * accept[i]
        remaining[p] *= 1.0 - accept[i]
        total += reach[i]
    return float(1.0 + total)
