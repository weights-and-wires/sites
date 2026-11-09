import numpy as np
from submission import cec_adder, make_adding_sequence

T = 100


def many(n=300, T=T):
    rng = np.random.default_rng(7)
    return [make_adding_sequence(rng, T) for _ in range(n)]


def test_shape_and_length_range():
    lengths = set()
    for X, _ in many():
        assert X.ndim == 2 and X.shape[1] == 2, f"expected shape (L, 2), got {X.shape}"
        lengths.add(len(X))
    assert min(lengths) >= T and max(lengths) <= T + T // 10, f"lengths must lie in [{T}, {T + T // 10}], got [{min(lengths)}, {max(lengths)}]"
    assert len(lengths) > 3, f"the length must be random, but only saw {sorted(lengths)}"
    assert max(lengths) == T + T // 10, f"the upper end {T + T // 10} must be reachable, max seen was {max(lengths)}"


def test_values_in_range():
    for X, _ in many(50):
        assert np.all(np.abs(X[:, 0]) <= 1.0), f"values must lie in [-1, 1], got range [{X[:, 0].min()}, {X[:, 0].max()}]"
    allv = np.concatenate([X[:, 0] for X, _ in many(50)])
    assert allv.min() < -0.9 and allv.max() > 0.9, f"values must span the interval, got [{allv.min()}, {allv.max()}]"


def test_exactly_two_marked_in_allowed_places():
    for X, _ in many():
        marked = np.flatnonzero(X[:, 1] == 1.0)
        assert len(marked) == 2, f"expected exactly 2 pairs with marker 1.0, got {len(marked)} at {marked}"
        assert marked.max() < T // 2 - 1, f"both marked pairs must be in the first {T // 2 - 1}, got {marked}"
        assert marked.min() < 10, f"one marked pair must be among the first 10, got {marked}"


def test_marker_values_elsewhere():
    for X, _ in many():
        assert set(np.unique(X[:, 1])) <= {-1.0, 0.0, 1.0}, f"markers must be -1, 0, or 1, got {np.unique(X[:, 1])}"
        assert X[-1, 1] == -1.0, f"the last pair must have marker -1.0, got {X[-1, 1]}"
        if X[0, 1] != 1.0:
            assert X[0, 1] == -1.0, f"the first pair must have marker -1.0 unless marked, got {X[0, 1]}"
        mid = X[1:-1, 1]
        assert np.sum(mid == -1.0) == 0, f"only the first and last pairs may carry -1.0, got {np.flatnonzero(mid == -1.0) + 1}"


def test_marked_first_pair_is_zeroed():
    seen = 0
    for X, _ in many(600):
        if X[0, 1] == 1.0:
            seen += 1
            assert X[0, 0] == 0.0, f"a marked first pair must have value 0.0, got {X[0, 0]}"
    assert seen > 0, "the generator never marked the first pair in 600 sequences; index 0 must be possible"


def test_target_formula():
    for X, target in many():
        marked = X[:, 1] == 1.0
        want = 0.5 + X[marked, 0].sum() / 4.0
        assert abs(target - want) < 1e-12, f"expected target {want}, got {target}"
        assert 0.0 <= target <= 1.0, f"target must lie in [0, 1], got {target}"


def test_reproducible_with_seed():
    X1, t1 = make_adding_sequence(np.random.default_rng(3), T)
    X2, t2 = make_adding_sequence(np.random.default_rng(3), T)
    assert np.array_equal(X1, X2) and t1 == t2, "the same seed must give the same sequence"


def test_cec_adder_matches_target():
    for X, target in many():
        got = cec_adder(X)
        assert abs(got - target) < 1e-9, f"expected {target}, got {got}"


def test_cec_adder_hand_example():
    X = np.array([[0.9, -1.0], [0.25, 1.0], [-0.8, 0.0], [0.5, 1.0], [0.3, -1.0]])
    got = cec_adder(X)
    assert abs(got - 0.6875) < 1e-12, f"expected 0.5 + (0.25 + 0.5)/4 = 0.6875, got {got}"
