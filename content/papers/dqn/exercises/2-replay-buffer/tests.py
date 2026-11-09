import numpy as np
from submission import ReplayBuffer


def fill(buf, n):
    """Push transition i as s = [i, i], a = i % 4, r = i / 10, s2 = [i + 1, i + 1], done = (i % 5 == 0)."""
    for i in range(n):
        buf.push(np.array([i, i], dtype=float), i % 4, i / 10, np.array([i + 1, i + 1], dtype=float), float(i % 5 == 0))


def test_length_grows_then_caps():
    buf = ReplayBuffer(5, 2)
    assert len(buf) == 0, f"an empty memory has length 0, got {len(buf)}"
    fill(buf, 3)
    assert len(buf) == 3, f"expected 3 after 3 pushes, got {len(buf)}"
    fill(buf, 10)
    assert len(buf) == 5, f"expected the capacity 5 after 13 pushes, got {len(buf)}"


def test_sample_shapes():
    buf = ReplayBuffer(10, 2)
    fill(buf, 10)
    S, A, R, S2, D = buf.sample(7, np.random.default_rng(0))
    assert np.shape(S) == (7, 2), f"S: expected (7, 2), got {np.shape(S)}"
    assert np.shape(S2) == (7, 2), f"S2: expected (7, 2), got {np.shape(S2)}"
    for name, x in (("A", A), ("R", R), ("D", D)):
        assert np.shape(x) == (7,), f"{name}: expected (7,), got {np.shape(x)}"


def test_columns_stay_aligned():
    buf = ReplayBuffer(20, 2)
    fill(buf, 20)
    S, A, R, S2, D = buf.sample(200, np.random.default_rng(1))
    i = S[:, 0].astype(int)
    assert np.array_equal(S[:, 1].astype(int), i), "both state coordinates must come from the same transition"
    assert np.array_equal(A, i % 4), f"actions must match their states; got A={A[:5]} for i={i[:5]}"
    assert np.allclose(R, i / 10), f"rewards must match their states; got R={R[:5]} for i={i[:5]}"
    assert np.array_equal(S2[:, 0].astype(int), i + 1), f"next states must match; got {S2[:5, 0]} for i={i[:5]}"
    assert np.array_equal(D, (i % 5 == 0).astype(float)), f"done flags must match; got {D[:5]} for i={i[:5]}"


def test_overwrites_the_oldest():
    buf = ReplayBuffer(5, 2)
    fill(buf, 8)  # transitions 0..7; the memory keeps 3..7
    S, *_ = buf.sample(2000, np.random.default_rng(2))
    seen = set(S[:, 0].astype(int).tolist())
    assert seen == {3, 4, 5, 6, 7}, f"expected the 5 newest transitions {{3, 4, 5, 6, 7}}, got {sorted(seen)}"


def test_partial_memory_never_samples_empty_slots():
    buf = ReplayBuffer(100, 2)
    fill(buf, 4)
    S, A, R, S2, D = buf.sample(500, np.random.default_rng(3))
    seen = set(S[:, 0].astype(int).tolist())
    assert seen == {0, 1, 2, 3}, f"expected only the 4 stored transitions {{0, 1, 2, 3}}, got {sorted(seen)}"
    assert np.all(S2[:, 0] >= 1), "a zero next state means an empty slot was sampled"


def test_sampling_is_uniform():
    buf = ReplayBuffer(8, 2)
    fill(buf, 8)
    S, *_ = buf.sample(16000, np.random.default_rng(4))
    counts = np.bincount(S[:, 0].astype(int), minlength=8)
    assert counts.min() > 1700 and counts.max() < 2300, f"expected about 2000 draws per transition, got {counts}"


def test_stored_transition_is_a_copy():
    buf = ReplayBuffer(3, 2)
    s = np.array([1.0, 1.0])
    buf.push(s, 0, 0.0, s, 0.0)
    s[:] = 99.0
    S, *_ = buf.sample(1, np.random.default_rng(5))
    assert np.allclose(S, [[1.0, 1.0]]), f"changing the array after push must not change the memory, got {S}"


def test_sampling_is_reproducible_from_the_rng():
    buf = ReplayBuffer(10, 2)
    fill(buf, 10)
    a = buf.sample(5, np.random.default_rng(7))[0]
    b = buf.sample(5, np.random.default_rng(7))[0]
    assert np.array_equal(a, b), "the same seed must give the same minibatch"
