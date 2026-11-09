import numpy as np
from submission import ema_step, usage


def test_ema_update_by_hand():
    cb = np.array([[0.0, 0.0], [10.0, 10.0]])
    batch = np.array([[1.0, 1.0], [3.0, 1.0]])  # both rows are nearest to entry 0
    rng = np.random.default_rng(0)
    new_cb, cnt, sm, idx = ema_step(cb, np.array([2.0, 2.0]), np.array([[0.0, 0.0], [20.0, 20.0]]), batch, rng, decay=0.9, threshold=0.0)
    assert idx.tolist() == [0, 0], f"both rows are nearest to entry 0; expected idx [0, 0], got {idx.tolist()}"
    want_cnt = np.array([0.9 * 2 + 0.1 * 2, 0.9 * 2])
    assert np.allclose(cnt, want_cnt), f"expected ema_count {want_cnt}, got {cnt}"
    want_sum = np.array([[0.1 * 4.0, 0.1 * 2.0], [18.0, 18.0]])
    assert np.allclose(sm, want_sum), f"expected ema_sum {want_sum.tolist()}, got {sm.tolist()}"
    want_cb = want_sum / want_cnt[:, None]
    assert np.allclose(new_cb, want_cb), f"expected codebook {want_cb.tolist()}, got {new_cb.tolist()}"


def test_inputs_not_modified():
    cb = np.array([[0.0, 0.0], [5.0, 5.0]])
    cnt = np.ones(2)
    sm = cb.copy()
    batch = np.array([[0.1, 0.1], [5.1, 5.0], [0.0, 0.2]])
    ema_step(cb, cnt, sm, batch, np.random.default_rng(0))
    assert np.array_equal(cb, [[0.0, 0.0], [5.0, 5.0]]), f"codebook argument was changed: {cb.tolist()}"
    assert np.array_equal(cnt, np.ones(2)), f"ema_count argument was changed: {cnt.tolist()}"
    assert np.array_equal(sm, [[0.0, 0.0], [5.0, 5.0]]), f"ema_sum argument was changed: {sm.tolist()}"


def test_dead_entry_is_replaced_from_the_batch():
    cb = np.array([[0.0, 0.0], [100.0, 100.0]])
    batch = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    new_cb, cnt, sm, idx = ema_step(cb, np.ones(2), cb.copy(), batch, np.random.default_rng(3), decay=0.5, threshold=0.6)
    # Entry 1 gets no rows: its count falls to 0.5, which is below the threshold of 0.6.
    assert idx.tolist() == [0, 0, 0, 0], f"all rows are nearest to entry 0; got {idx.tolist()}"
    assert any(np.allclose(new_cb[1], row) for row in batch), f"dead entry 1 must become a row of the batch; got {new_cb[1].tolist()}"
    assert cnt[1] == 1.0, f"a replaced entry restarts with ema_count 1; got {cnt[1]}"
    assert np.allclose(sm[1], new_cb[1]), f"a replaced entry restarts with ema_sum equal to its vector; got {sm[1].tolist()} and {new_cb[1].tolist()}"
    want0 = (0.5 * np.array([0.0, 0.0]) + 0.5 * batch.sum(0)) / (0.5 * 1 + 0.5 * 4)
    assert np.allclose(new_cb[0], want0), f"live entry 0 should be ema_sum/ema_count = {want0.tolist()}, got {new_cb[0].tolist()}"


def test_live_entries_are_not_replaced():
    cb = np.array([[0.0, 0.0], [10.0, 0.0]])
    batch = np.array([[0.0, 0.1], [10.0, 0.1]])
    new_cb, cnt, sm, idx = ema_step(cb, np.ones(2), cb.copy(), batch, np.random.default_rng(0), decay=0.5, threshold=0.5)
    # count = 0.5 * 1 + 0.5 * 1 = 1 for both entries, so each entry is 0.5 * old_sum + 0.5 * batch row.
    want = 0.5 * cb + 0.5 * batch
    assert np.allclose(new_cb, want), f"both entries are used, so none is replaced; expected {want.tolist()}, got {new_cb.tolist()}"


def test_usage_and_perplexity():
    cb = np.array([[0.0], [10.0], [20.0], [30.0]])
    x = np.array([[0.1], [-0.2], [10.1], [9.9]])
    n, perp = usage(cb, x)
    assert n == 2, f"two of four entries are nearest to some row; expected 2, got {n}"
    assert np.isclose(perp, 2.0), f"two entries with equal frequency give perplexity 2; got {perp}"
    x = np.array([[0.0], [10.0], [20.0], [30.0]] * 3)
    n, perp = usage(cb, x)
    assert n == 4 and np.isclose(perp, 4.0), f"uniform use of 4 entries: expected (4, 4.0), got ({n}, {perp})"
    x = np.array([[0.0]] * 3 + [[10.0]])
    n, perp = usage(cb, x)
    want = np.exp(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
    assert np.isclose(perp, want), f"frequencies 0.75 and 0.25: expected perplexity {want:.4f}, got {perp}"


def test_replacement_rescues_a_bad_initialization():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [6, 0], [0, 6], [6, 6]], float)

    def data(n):
        return centers[rng.integers(0, 4, n)] + rng.normal(0, 0.3, (n, 2))

    cb = rng.normal(0, 40.0, (16, 2))  # far from the data
    cnt, sm = np.ones(16), cb.copy()
    for _ in range(200):
        cb, cnt, sm, _ = ema_step(cb, cnt, sm, data(256), rng)
    n_used, perp = usage(cb, data(2000))
    assert n_used >= 12, f"after 200 steps with dead-entry replacement at least 12 of 16 entries should be used; got {n_used}"
    assert perp > 8, f"perplexity should be well above 8 after training; got {perp:.2f}"


def test_dead_entries_get_different_vectors():
    rng = np.random.default_rng(5)
    cb = np.vstack([[0.0, 0.0], np.full((4, 2), 500.0)])
    batch = rng.normal(0, 1.0, (50, 2))
    new_cb, cnt, sm, idx = ema_step(cb, np.ones(5), cb.copy(), batch, np.random.default_rng(1), decay=0.5, threshold=0.6)
    dead = new_cb[1:]
    distinct = len({tuple(np.round(v, 9)) for v in dead})
    assert distinct >= 3, f"four dead entries should draw from the batch with rng.integers, giving at least 3 distinct vectors; got {distinct}"
    assert all(any(np.allclose(v, row) for row in batch) for v in dead), "every replacement must be a row of the batch"
