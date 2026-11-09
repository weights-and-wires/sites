import numpy as np
from submission import zero1_step, shard_slices, adam_update


def make(nd, psi, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=psi)
    sl = shard_slices(psi, nd)
    m = [np.zeros(s.stop - s.start) for s in sl]
    v = [np.zeros(s.stop - s.start) for s in sl]
    return rng, p, m, v


def reference(p, grads_per_step, lr=0.1):
    m, v = np.zeros_like(p), np.zeros_like(p)
    for t, wg in enumerate(grads_per_step, start=1):
        p, m, v = adam_update(p, wg.mean(axis=0), m, v, t, lr)
    return p, m, v


def test_one_step_matches_ddp():
    rng, p, m, v = make(4, 10)
    wg = rng.normal(size=(4, 10))
    got = zero1_step(p, wg, m, v, 1)
    want, _, _ = reference(p, [wg])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_three_steps_match_ddp():
    rng, p, m, v = make(4, 10)
    steps = [rng.normal(size=(4, 10)) for _ in range(3)]
    got = p
    for t, wg in enumerate(steps, start=1):
        got = zero1_step(got, wg, m, v, t)
    want, wm, wv = reference(p, steps)
    assert np.allclose(got, want), f"after 3 steps expected {want}, got {got}"
    assert np.allclose(np.concatenate(m), wm), f"concatenated m shards should equal DDP's m {wm}, got {np.concatenate(m)}"
    assert np.allclose(np.concatenate(v), wv), f"concatenated v shards should equal DDP's v {wv}, got {np.concatenate(v)}"


def test_uses_average_not_one_worker():
    rng, p, m, v = make(4, 8)
    wg = rng.normal(size=(4, 8))
    got = zero1_step(p, wg, m, v, 1)
    one_worker, _, _ = adam_update(p, wg[0], np.zeros(8), np.zeros(8), 1)
    assert not np.allclose(got, one_worker), "the update must use the gradient averaged over all workers, not worker 0's gradient"


def test_state_stays_sharded():
    rng, p, m, v = make(4, 10)
    zero1_step(p, rng.normal(size=(4, 10)), m, v, 1)
    assert [len(x) for x in m] == [3, 3, 2, 2], f"m shard lengths should stay [3, 3, 2, 2], got {[len(x) for x in m]}"
    assert [len(x) for x in v] == [3, 3, 2, 2], f"v shard lengths should stay [3, 3, 2, 2], got {[len(x) for x in v]}"


def test_state_is_updated_in_place():
    rng, p, m, v = make(2, 6)
    zero1_step(p, rng.normal(size=(2, 6)), m, v, 1)
    assert all(np.any(x != 0) for x in m), f"m shards should be nonzero after a step, got {m}"
    assert all(np.any(x != 0) for x in v), f"v shards should be nonzero after a step, got {v}"


def test_each_shard_sees_its_own_gradient_slice():
    # Gradient is zero everywhere except slice 2; only shard 2 may get nonzero state.
    nd, psi = 4, 8
    _, p, m, v = make(nd, psi)
    wg = np.zeros((nd, psi))
    wg[:, 4:6] = 1.0
    zero1_step(p, wg, m, v, 1)
    nz = [bool(np.any(x != 0)) for x in m]
    assert nz == [False, False, True, False], f"only shard 2 should hold nonzero m, got {nz}"


def test_does_not_modify_input_params():
    rng, p, m, v = make(4, 10)
    before = p.copy()
    zero1_step(p, rng.normal(size=(4, 10)), m, v, 1)
    assert np.array_equal(p, before), "zero1_step should return new parameters and leave the input array unchanged"


def test_uneven_split_covers_every_parameter():
    rng, p, m, v = make(3, 10)
    wg = rng.normal(size=(3, 10))
    got = zero1_step(p, wg, m, v, 1)
    want, _, _ = reference(p, [wg])
    assert got.shape == (10,), f"expected shape (10,), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_bias_correction_uses_step_number():
    rng, p, m, v = make(2, 6)
    wg = rng.normal(size=(2, 6))
    a = zero1_step(p, wg, [x.copy() for x in m], [x.copy() for x in v], 1)
    b = zero1_step(p, wg, [x.copy() for x in m], [x.copy() for x in v], 5)
    assert not np.allclose(a, b), "step t=1 and t=5 should give different updates because of bias correction"
