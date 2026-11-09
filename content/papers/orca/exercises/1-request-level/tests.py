from submission import static_batching


def test_page_example():
    fin, use = static_batching([0, 0, 1, 2], [2, 6, 2, 2], 2)
    assert list(fin) == [6, 6, 8, 8], f"expected finish [6, 6, 8, 8], got {list(fin)}"
    assert list(use) == [2, 2, 1, 1, 1, 1, 2, 2], f"expected useful [2, 2, 1, 1, 1, 1, 2, 2], got {list(use)}"


def test_short_request_waits_for_long_one():
    fin, _ = static_batching([0, 0], [1, 5], 2)
    assert list(fin) == [5, 5], f"the 1-iteration request must wait for the batch; expected [5, 5], got {list(fin)}"


def test_late_arrival_waits_for_next_batch():
    fin, _ = static_batching([0, 1], [4, 2], 2)
    assert list(fin) == [4, 6], f"request 1 arrives at t=1 and must wait until t=4; expected [4, 6], got {list(fin)}"


def test_idle_gap():
    fin, use = static_batching([0, 10], [2, 3], 4)
    assert list(fin) == [2, 13], f"expected finish [2, 13], got {list(fin)}"
    assert len(use) == 13 and use[2:10] == [0] * 8, f"iterations 2..9 are idle and must count 0; got {list(use)}"


def test_max_batch_size_respected():
    fin, use = static_batching([0, 0, 0], [3, 3, 3], 2)
    assert list(fin) == [3, 3, 6], f"expected [3, 3, 6], got {list(fin)}"
    assert max(use) == 2, f"no iteration may run more than max_bs=2 requests; got {list(use)}"


def test_fcfs_by_arrival_not_index():
    fin, _ = static_batching([5, 0], [1, 1], 1)
    assert list(fin) == [6, 1], f"request 1 arrives first and must run first; expected [6, 1], got {list(fin)}"


def test_useful_work_equals_total_tokens():
    arr, lens = [0, 0, 3, 3, 9], [4, 1, 5, 2, 3]
    _, use = static_batching(arr, lens, 3)
    assert sum(use) == sum(lens), f"useful slots must equal total tokens {sum(lens)}, got {sum(use)}"
