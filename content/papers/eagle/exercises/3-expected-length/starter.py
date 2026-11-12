import numpy as np


def expected_length(parents, accept):
    """Expected tokens per round: 1 + sum of P(node accepted).

    parents: parent index of each node, -1 for a child of the last context token.
    accept:  P(node accepted | it is tested) for each node.
    """
    raise NotImplementedError
