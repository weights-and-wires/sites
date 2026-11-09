import itertools

import numpy as np

from submission import best_path_decode, path_probability

Y = np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])  # columns: blank, a, b


def collapse(path):
    out, prev = [], None
    for s in path:
        if s != prev and s != 0:
            out.append(s)
        prev = s
    return tuple(out)


def label_probability(y, label):
    return sum(
        path_probability(y, path)
        for path in itertools.product(range(y.shape[1]), repeat=y.shape[0])
        if collapse(path) == tuple(label)
    )


def test_page_example():
    got = best_path_decode(Y)
    assert got == (1,), f"argmax path is a a -, so B = (1,); got {got}"


def test_argmax_axis_is_symbols():
    # Each frame's winner is a different symbol; the wrong axis would pick one frame per symbol.
    y = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    got = best_path_decode(y)
    assert got == (1, 2, 1), f"path a b - a: expected (1, 2, 1), got {got}"


def test_repeats_merge_and_blank_separates():
    y_aa = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])  # a a - a
    got = best_path_decode(y_aa)
    assert got == (1, 1), f"path a a - a: expected (1, 1), got {got}"
    y_a = np.array([[0.1, 0.9]] * 3)  # a a a
    got = best_path_decode(y_a)
    assert got == (1,), f"path a a a: expected (1,), got {got}"


def test_all_blank_gives_empty_labelling():
    y = np.array([[0.9, 0.1]] * 4)
    got = best_path_decode(y)
    assert got == (), f"all-blank path: expected (), got {got}"


def test_nonzero_blank_index():
    y = np.array([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])  # blank = 2
    got = best_path_decode(y, blank=2)
    assert got == (0, 1), f"blank=2, path - a b ... expected (0, 1), got {got}"


def test_path_probability():
    got = path_probability(Y, (1, 1, 0))
    assert abs(got - 0.5 * 0.4 * 0.5) < 1e-12, f"p(a a -|x): expected 0.1, got {got}"


def test_best_path_is_not_always_the_best_labelling():
    # Blank wins each frame (0.36 for '--'), but 'a' collects aa, a-, -a = 0.64 in total.
    y = np.array([[0.6, 0.4], [0.6, 0.4]])
    got = best_path_decode(y)
    assert got == (), f"expected the empty labelling, got {got}"
    p_decoded = label_probability(y, got)
    p_best = label_probability(y, (1,))
    assert abs(p_decoded - 0.36) < 1e-12, f"p(empty|x): expected 0.36, got {p_decoded}"
    assert abs(p_best - 0.64) < 1e-12, f"p(a|x): expected 0.64, got {p_best}"
