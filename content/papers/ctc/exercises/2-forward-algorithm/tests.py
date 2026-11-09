import itertools

import numpy as np

from submission import ctc_forward

Y = np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])  # columns: blank, a, b


def collapse(path):
    out, prev = [], None
    for s in path:
        if s != prev and s != 0:
            out.append(s)
        prev = s
    return tuple(out)


def brute(y, label):
    T, K = y.shape
    return sum(
        float(np.prod([y[t, k] for t, k in enumerate(path)]))
        for path in itertools.product(range(K), repeat=T)
        if collapse(path) == tuple(label)
    )


def random_y(seed, T, K):
    rng = np.random.default_rng(seed)
    y = rng.random((T, K)) + 0.05
    return y / y.sum(axis=1, keepdims=True)


def test_alpha_table_for_ab():
    alpha, p = ctc_forward(Y, (1, 2))
    want = np.array([
        [0.4, 0.5, 0.0, 0.0, 0.0],
        [0.12, 0.36, 0.15, 0.15, 0.0],
        [0.06, 0.096, 0.255, 0.198, 0.075],
    ])
    assert alpha.shape == (3, 5), f"alpha shape: expected (3, 5), got {alpha.shape}"
    assert np.allclose(alpha, want), f"alpha: expected\n{want}\ngot\n{alpha}"
    assert abs(p - 0.273) < 1e-9, f"p(ab|x): expected 0.273, got {p}"


def test_repeated_label_blocks_the_skip():
    alpha, p = ctc_forward(Y, (1, 1))
    want = np.array([
        [0.4, 0.5, 0.0, 0.0, 0.0],
        [0.12, 0.36, 0.15, 0.0, 0.0],
        [0.06, 0.096, 0.255, 0.03, 0.0],
    ])
    assert np.allclose(alpha, want), f"alpha for 'aa': expected\n{want}\ngot\n{alpha}"
    assert abs(p - 0.03) < 1e-9, f"p(aa|x): expected 0.03 (only a-a), got {p}"


def test_matches_brute_force_on_random_inputs():
    for seed, (T, K, label) in enumerate([(5, 3, (1, 2)), (6, 3, (2, 1, 2)), (5, 4, (3, 1, 1)), (6, 3, (1,)), (6, 4, (1, 2, 3)), (7, 3, (1, 1, 1))]):
        y = random_y(seed, T, K)
        _, p = ctc_forward(y, label)
        want = brute(y, label)
        assert abs(p - want) < 1e-12, f"T={T} K={K} label={label}: expected {want}, got {p}"


def test_empty_label():
    y = random_y(7, 4, 3)
    alpha, p = ctc_forward(y, ())
    want = float(np.prod(y[:, 0]))
    assert alpha.shape == (4, 1), f"alpha shape for the empty label: expected (4, 1), got {alpha.shape}"
    assert abs(p - want) < 1e-12, f"p(empty|x): expected {want} (all blanks), got {p}"


def test_label_too_long_has_zero_probability():
    y = random_y(8, 3, 3)
    _, p = ctc_forward(y, (1, 2, 1, 2))
    assert p == 0.0, f"four labels cannot fit in T=3, expected 0.0, got {p}"
    _, p = ctc_forward(y, (1, 1, 1))
    assert p == 0.0, f"'aaa' needs T >= 5, expected 0.0, got {p}"


def test_all_labellings_sum_to_one():
    y = random_y(9, 5, 3)
    labellings = {()}
    for n in range(1, 6):
        labellings |= set(itertools.product([1, 2], repeat=n))
    total = sum(ctc_forward(y, l)[1] for l in labellings)
    assert abs(total - 1.0) < 1e-12, f"labelling probabilities must sum to 1, got {total}"


def test_path_counts_with_all_ones():
    # With y = 1 everywhere, p is the number of paths in B^-1(l). For l = 'ab' and T = 3 there are 5.
    _, p = ctc_forward(np.ones((3, 3)), (1, 2))
    assert p == 5.0, f"paths for 'ab' with T=3: expected 5, got {p}"
    _, p = ctc_forward(np.ones((10, 4)), (1, 2, 3))
    assert p == 1716.0, f"paths for 'abc' with T=10: expected 1716, got {p}"
