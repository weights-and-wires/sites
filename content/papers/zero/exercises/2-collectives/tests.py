import numpy as np
from submission import reduce_scatter, all_gather, all_reduce, comm_volume

XS = np.arange(24, dtype=float).reshape(4, 6)  # 4 workers, 6 elements


def test_reduce_scatter_shards():
    got = reduce_scatter(XS)
    total = XS.sum(axis=0)
    assert len(got) == 4, f"expected 4 shards, got {len(got)}"
    for i, shard in enumerate(got):
        want = np.array_split(total, 4)[i]
        assert np.allclose(shard, want), f"shard {i}: expected {want}, got {shard}"


def test_reduce_scatter_sums_not_means():
    got = reduce_scatter(np.ones((4, 8)))
    assert np.allclose(np.concatenate(got), 4.0), f"each element should be the sum 4.0, got {np.concatenate(got)}"


def test_reduce_scatter_uneven():
    xs = np.arange(30, dtype=float).reshape(3, 10)
    got = reduce_scatter(xs)
    assert [len(s) for s in got] == [4, 3, 3], f"expected shard lengths [4, 3, 3], got {[len(s) for s in got]}"
    assert np.allclose(np.concatenate(got), xs.sum(axis=0)), "shards must cover the full sum in order"


def test_all_gather():
    shards = [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0])]
    got = all_gather(shards)
    assert len(got) == 3, f"expected one copy per worker (3), got {len(got)}"
    for i, g in enumerate(got):
        assert np.allclose(g, [1, 2, 3, 4, 5]), f"worker {i}: expected [1 2 3 4 5], got {g}"


def test_all_reduce_matches_sum():
    got = all_reduce(XS)
    want = XS.sum(axis=0)
    assert len(got) == 4, f"expected 4 copies, got {len(got)}"
    for i, g in enumerate(got):
        assert np.allclose(g, want), f"worker {i}: expected {want}, got {g}"


def test_all_reduce_random():
    rng = np.random.default_rng(1)
    xs = rng.normal(size=(5, 13))
    for g in all_reduce(xs):
        assert np.allclose(g, xs.sum(axis=0)), "every worker must hold the same elementwise sum"


def test_volume_data_parallel():
    assert comm_volume(0) == 2, f"data parallelism moves 2 Psi, got {comm_volume(0)}"


def test_volume_stage_1_and_2_match_ddp():
    assert comm_volume(1) == comm_volume(0), f"stage 1 should equal DDP (2), got {comm_volume(1)}"
    assert comm_volume(2) == comm_volume(0), f"stage 2 should equal DDP (2), got {comm_volume(2)}"


def test_volume_stage_3():
    assert comm_volume(3) == 3, f"stage 3 moves 3 Psi, got {comm_volume(3)}"
    assert comm_volume(3) / comm_volume(0) == 1.5, f"expected 1.5x the baseline, got {comm_volume(3) / comm_volume(0)}"
