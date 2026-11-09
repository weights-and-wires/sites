import numpy as np
from submission import additive_attention

# The example on the page (scripts/additive_example.py).
H0 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Wa0 = np.array([[1.0]])
Ua0 = np.array([[1.0, -1.0]])
va0 = np.array([2.0])


def random_problem(seed, n=4, n_prime=6, Tx=5):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n), rng.normal(size=(Tx, 2 * n)), rng.normal(size=(n_prime, n)),
            rng.normal(size=(n_prime, 2 * n)), rng.normal(size=n_prime))


def test_tiny_example_by_hand():
    c, alpha = additive_attention(np.array([0.0]), H0, Wa0, Ua0, va0)
    want = [0.7902, 0.0376, 0.1723]
    assert np.allclose(alpha, want, atol=1e-4), f"expected alpha {want}, got {alpha}"
    assert np.allclose(c, [0.9624, 0.2098], atol=1e-4), f"expected c [0.9624, 0.2098], got {c}"


def test_state_changes_the_weights():
    c, alpha = additive_attention(np.array([1.0]), H0, Wa0, Ua0, va0)
    want = [0.5517, 0.0802, 0.3680]
    assert np.allclose(alpha, want, atol=1e-4), f"with s = 1 expected alpha {want}, got {alpha}"


def test_shapes_and_normalization():
    s, H, Wa, Ua, va = random_problem(0)
    c, alpha = additive_attention(s, H, Wa, Ua, va)
    assert alpha.shape == (5,), f"alpha shape: expected (5,), got {alpha.shape}"
    assert c.shape == (8,), f"c shape: expected (8,), got {c.shape}"
    assert np.isclose(alpha.sum(), 1.0), f"weights must sum to 1, got {alpha.sum()}"
    assert np.all(alpha > 0), f"softmax weights must be positive, got {alpha}"


def test_context_is_weighted_sum_of_annotations():
    s, H, Wa, Ua, va = random_problem(1)
    c, alpha = additive_attention(s, H, Wa, Ua, va)
    want = sum(alpha[j] * H[j] for j in range(H.shape[0]))
    assert np.allclose(c, want), f"c must equal sum_j alpha_j h_j\nexpected {want}\ngot {c}"


def test_matches_loop_reference():
    s, H, Wa, Ua, va = random_problem(2)
    e = np.array([va @ np.tanh(Wa @ s + Ua @ H[j]) for j in range(H.shape[0])])
    want = np.exp(e - e.max())
    want /= want.sum()
    _, alpha = additive_attention(s, H, Wa, Ua, va)
    assert np.allclose(alpha, want), f"expected {want}, got {alpha}"


def test_weights_depend_on_the_decoder_state():
    # Without the tanh, the s term is the same for every j and cancels in the softmax.
    _, H, Wa, Ua, va = random_problem(3)
    rng = np.random.default_rng(30)
    _, a1 = additive_attention(rng.normal(size=4), H, Wa, Ua, va)
    _, a2 = additive_attention(rng.normal(size=4), H, Wa, Ua, va)
    assert not np.allclose(a1, a2, atol=1e-3), f"two different states gave the same weights {a1}; keep the tanh around W_a s + U_a h_j"


def test_permuting_annotations_permutes_weights():
    s, H, Wa, Ua, va = random_problem(4)
    perm = np.array([3, 0, 4, 1, 2])
    c1, a1 = additive_attention(s, H, Wa, Ua, va)
    c2, a2 = additive_attention(s, H[perm], Wa, Ua, va)
    assert np.allclose(a2, a1[perm]), f"expected {a1[perm]}, got {a2}"
    assert np.allclose(c1, c2), "the context must not change when the annotations are reordered"
