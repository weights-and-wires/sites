import numpy as np
from submission import lora_forward, merge, unmerge, swap

rng = np.random.default_rng(0)
d, k = 5, 7
W0 = rng.normal(size=(d, k))
X = rng.normal(size=(9, k))
A = rng.normal(size=(3, k))
B = rng.normal(size=(d, 3))
A2 = rng.normal(size=(2, k))
B2 = rng.normal(size=(d, 2))


def test_page_example():
    w0 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    got = merge(w0, np.array([[1.0, 0.0, -1.0]]), np.array([[1.0], [2.0]]), alpha=1.0)
    want = np.array([[2.0, 0.0, 1.0], [2.0, 1.0, -1.0]])
    assert np.allclose(got, want), f"expected\n{want}\ngot\n{got}"


def test_merged_output_equals_unmerged():
    for alpha in (1.0, 3.0, 12.0):
        W = merge(W0, A, B, alpha)
        assert W.shape == (d, k), f"merged weight expected shape {(d, k)}, got {W.shape}"
        want = lora_forward(X, W0, A, B, alpha)
        got = X @ W.T
        assert np.allclose(got, want), f"alpha={alpha}: merged output differs from unmerged by {np.abs(got - want).max():.3g}"


def test_merge_scales_by_alpha_over_r():
    r = A.shape[0]
    got = merge(W0, A, B, alpha=6.0) - W0
    want = 2.0 * (B @ A)
    assert r == 3 and np.allclose(got, want), f"alpha/r = 2, so the update should be 2 B A; max diff {np.abs(got - want).max():.3g}"


def test_merge_does_not_modify_its_input():
    before = W0.copy()
    merge(W0, A, B, alpha=3.0)
    assert np.array_equal(W0, before), "merge must return a new array and leave W0 unchanged"


def test_unmerge_recovers_the_base():
    W = merge(W0, A, B, alpha=5.0)
    back = unmerge(W, A, B, alpha=5.0)
    assert np.allclose(back, W0), f"unmerge must recover W0; max diff {np.abs(back - W0).max():.3g}"


def test_swap_equals_fresh_merge():
    alpha = 4.0
    W = merge(W0, A, B, alpha)
    got = swap(W, (A, B), (A2, B2), alpha)
    want = merge(W0, A2, B2, alpha)
    assert np.allclose(got, want), f"after the swap the weight must equal W0 plus the new adapter; max diff {np.abs(got - want).max():.3g}"
    out = X @ got.T
    assert np.allclose(out, lora_forward(X, W0, A2, B2, alpha)), "swapped weight must match the new adapter's unmerged output"
