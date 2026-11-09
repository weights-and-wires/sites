import numpy as np
from submission import switch_route

# The router logits from the page: 6 tokens, 3 experts. Using x = h and an
# identity Wr makes the logits equal to h.
H = np.array([
    [2.0, 0.0, 0.0],
    [1.5, 0.5, 0.0],
    [1.0, 0.0, 0.5],
    [2.0, 1.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
])
EYE = np.eye(3)


def test_probabilities_match_equation_1():
    p, _, _ = switch_route(H, EYE)
    assert p.shape == (6, 3), f"expected shape (6, 3), got {p.shape}"
    assert np.allclose(p[0], [0.7870, 0.1065, 0.1065], atol=1e-4), f"row 0 expected [0.787 0.1065 0.1065], got {p[0]}"
    assert np.allclose(p[2], [0.5065, 0.1863, 0.3072], atol=1e-4), f"row 2 expected [0.5065 0.1863 0.3072], got {p[2]}"


def test_softmax_is_over_experts():
    # T = 5 tokens and N = 3 experts, so the wrong axis is a different shape of sum.
    rng = np.random.default_rng(1)
    x, Wr = rng.normal(size=(5, 4)), rng.normal(size=(4, 3))
    p, _, _ = switch_route(x, Wr)
    assert np.allclose(p.sum(axis=1), 1), f"each token's probabilities must sum to 1 over experts, got row sums {p.sum(axis=1)}"


def test_expert_choice():
    _, idx, _ = switch_route(H, EYE)
    assert np.array_equal(idx, [0, 0, 0, 0, 1, 2]), f"expected [0 0 0 0 1 2], got {idx}"


def test_gate_is_the_chosen_probability():
    p, idx, gate = switch_route(H, EYE)
    assert gate.shape == (6,), f"expected shape (6,), got {gate.shape}"
    want = np.array([0.7870, 0.6285, 0.5065, 0.6652, 0.7870, 0.7870])
    assert np.allclose(gate, want, atol=1e-4), f"expected {want}, got {gate}"
    assert np.allclose(gate, p.max(axis=1)), f"gate must equal the largest probability, got {gate} vs {p.max(axis=1)}"


def test_logits_use_x_times_wr():
    # A non-square Wr: d_model = 2, N = 3. Only x @ Wr has the right shape.
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    Wr = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    _, idx, _ = switch_route(x, Wr)
    assert np.array_equal(idx, [0, 2]), f"expected [0 2], got {idx}"


def test_large_logits_do_not_overflow():
    x = np.array([[1000.0, 0.0, 0.0], [0.0, 0.0, 1000.0]])
    p, idx, gate = switch_route(x, EYE)
    assert np.all(np.isfinite(p)), f"probabilities must be finite for logits of 1000, got {p}"
    assert np.array_equal(idx, [0, 2]), f"expected [0 2], got {idx}"
    assert np.allclose(gate, 1.0), f"expected gates of 1.0, got {gate}"
