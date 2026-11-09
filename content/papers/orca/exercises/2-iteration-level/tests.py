from submission import continuous_batching


def test_page_example():
    fin, use = continuous_batching([0, 0, 1, 2], [2, 6, 2, 2], 2)
    assert list(fin) == [2, 6, 4, 6], f"expected finish [2, 6, 4, 6], got {list(fin)}"
    assert list(use) == [2] * 6, f"expected useful [2, 2, 2, 2, 2, 2], got {list(use)}"


def test_short_request_returns_early():
    fin, _ = continuous_batching([0, 0], [1, 5], 2)
    assert list(fin) == [1, 5], f"the 1-iteration request must return at t=1; expected [1, 5], got {list(fin)}"


def test_late_arrival_joins_next_iteration():
    fin, _ = continuous_batching([0, 1], [4, 2], 2)
    assert list(fin) == [4, 3], f"request 1 joins at t=1 and finishes at 3; expected [4, 3], got {list(fin)}"


def test_freed_slot_is_refilled():
    fin, use = continuous_batching([0, 0, 0], [1, 4, 4], 2)
    assert list(fin) == [1, 4, 5], f"request 2 takes the slot freed at t=1; expected [1, 4, 5], got {list(fin)}"
    assert list(use) == [2, 2, 2, 2, 1], f"expected useful [2, 2, 2, 2, 1], got {list(use)}"


def test_idle_gap():
    fin, use = continuous_batching([0, 10], [2, 3], 4)
    assert list(fin) == [2, 13], f"expected finish [2, 13], got {list(fin)}"
    assert list(use[2:10]) == [0] * 8, f"iterations 2..9 are idle and must count 0; got {list(use)}"


def test_fcfs_by_arrival_not_index():
    fin, _ = continuous_batching([5, 0], [1, 1], 1)
    assert list(fin) == [6, 1], f"expected [6, 1], got {list(fin)}"
    fin, _ = continuous_batching([1, 0], [2, 3], 1)
    assert list(fin) == [5, 3], f"request 1 arrived first and keeps the single slot; expected [5, 3], got {list(fin)}"


def test_max_batch_size_and_fcfs_under_load():
    fin, use = continuous_batching([0, 0, 0], [3, 3, 3], 2)
    assert list(fin) == [3, 3, 6], f"requests 0 and 1 run first; expected [3, 3, 6], got {list(fin)}"
    assert max(use) <= 2, f"no iteration may run more than max_bs=2 requests; got {list(use)}"


def test_never_slower_than_request_level():
    arr = [0, 1, 1, 2, 5, 5, 7, 8]
    lens = [5, 1, 7, 2, 3, 6, 1, 4]
    fin, use = continuous_batching(arr, lens, 3)
    assert sum(use) == sum(lens), f"useful slots must equal total tokens {sum(lens)}, got {sum(use)}"
    assert max(fin) <= sum(lens) + max(arr), f"makespan {max(fin)} is too long for this workload"
    assert all(f - a >= l for f, a, l in zip(fin, arr, lens)), f"no request can finish faster than its length; finish {list(fin)}"
