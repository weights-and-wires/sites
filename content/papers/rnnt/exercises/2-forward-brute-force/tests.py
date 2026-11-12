import math

import numpy as np
from submission import brute_force, forward

# The toy from the page (T = 3, y = (a, b)).
EMIT = np.array([[0.6, 1 / 3], [0.2, 0.75], [0.25, 0.4]])
NULL = np.array([[0.2, 1 / 6, 0.5], [0.2, 0.125, 0.5], [0.5, 0.4, 0.8]])


def random_lattice(T, U, seed):
    rng = np.random.default_rng(seed)
    emit = rng.uniform(0.05, 0.6, size=(T, U))
    null = rng.uniform(0.05, 0.6, size=(T, U + 1))
    return emit, null


def test_alpha_table_from_the_page():
    alpha, _ = forward(EMIT, NULL)
    want = np.array([[1.0, 0.6, 0.2], [0.2, 0.14, 0.205], [0.04, 0.0275, 0.1135]])
    assert np.allclose(alpha, want, atol=1e-9), f"expected alpha\n{want}\ngot\n{alpha}"


def test_probability_from_the_page():
    _, prob = forward(EMIT, NULL)
    want = 0.1135 * 0.8
    assert abs(prob - want) < 1e-9, f"expected Pr(y|x) = alpha(T, U) * null(T, U) = {want}, got {prob}"


def test_one_node_by_hand():
    # alpha(2, 1) = alpha(1, 1) null(1, 1) + alpha(2, 0) y(2, 0)
    alpha, _ = forward(EMIT, NULL)
    want = 0.6 * (1 / 6) + 0.2 * 0.2
    assert abs(alpha[1, 1] - want) < 1e-9, f"expected alpha(2, 1) = {want}, got {alpha[1, 1]}"


def test_brute_force_matches_the_page():
    prob, n = brute_force(EMIT, NULL)
    assert n == 6, f"expected C(3 + 2 - 1, 2) = 6 alignments, got {n}"
    assert abs(prob - 0.1135 * 0.8) < 1e-9, f"expected {0.1135 * 0.8}, got {prob}"


def test_forward_equals_brute_force_on_random_lattices():
    for T, U, seed in [(1, 3, 0), (4, 2, 1), (3, 4, 2), (5, 1, 3), (2, 2, 4)]:
        emit, null = random_lattice(T, U, seed)
        _, fwd = forward(emit, null)
        bf, n = brute_force(emit, null)
        assert n == math.comb(T + U - 1, U), f"T={T}, U={U}: expected {math.comb(T + U - 1, U)} alignments, got {n}"
        assert abs(fwd - bf) < 1e-12, f"T={T}, U={U}: forward {fwd} and brute force {bf} disagree"


def test_single_frame_emits_every_label_in_that_frame():
    emit = np.array([[0.5, 0.25, 0.1]])
    null = np.array([[0.3, 0.3, 0.3, 0.4]])
    _, prob = forward(emit, null)
    want = 0.5 * 0.25 * 0.1 * 0.4
    assert abs(prob - want) < 1e-12, f"with one frame there is one alignment; expected {want}, got {prob}"
    assert abs(brute_force(emit, null)[0] - want) < 1e-12, "brute force disagrees on the single alignment"


def test_empty_target_is_the_product_of_nulls():
    null = np.array([[0.7], [0.6], [0.9]])
    emit = np.zeros((3, 0))
    alpha, prob = forward(emit, null)
    assert abs(prob - 0.7 * 0.6 * 0.9) < 1e-12, f"expected {0.7 * 0.6 * 0.9}, got {prob}"
    assert alpha.shape == (3, 1), f"expected alpha shape (3, 1), got {alpha.shape}"
    assert abs(brute_force(emit, null)[0] - prob) < 1e-12, "brute force disagrees for U = 0"


def test_final_null_is_included():
    emit, null = random_lattice(3, 2, 7)
    alpha, prob = forward(emit, null)
    assert abs(prob - alpha[-1, -1] * null[-1, -1]) < 1e-15, "Pr(y|x) must multiply alpha(T, U) by null(T, U)"
    assert prob < alpha[-1, -1], f"expected prob below alpha(T, U) = {alpha[-1, -1]}, got {prob}"
