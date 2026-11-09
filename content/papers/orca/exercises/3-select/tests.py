from submission import select


def req(arrival, state, max_tokens):
    return {"arrival": arrival, "state": state, "max_tokens": max_tokens}


PAGE_POOL = [req(0, "INCREMENT", 40), req(1, "INITIATION", 50), req(2, "INITIATION", 30), req(3, "INITIATION", 10)]


def test_page_example():
    batch, n = select(PAGE_POOL, 40, 4, 100)
    assert list(batch) == [0, 1], f"expected batch [0, 1], got {list(batch)}"
    assert n == 90, f"expected n_rsrv 90, got {n}"


def test_stops_instead_of_skipping():
    batch, _ = select(PAGE_POOL, 40, 4, 100)
    assert 3 not in batch, f"request 3 fits but sits behind request 2, which doesn't; FCFS must hold, got {list(batch)}"


def test_exact_fit_is_allowed():
    batch, n = select([req(0, "INITIATION", 60)], 40, 4, 100)
    assert list(batch) == [0] and n == 100, f"reserving exactly n_slots is allowed; got batch {list(batch)}, n_rsrv {n}"


def test_increment_reserves_nothing():
    pool = [req(0, "INCREMENT", 500), req(1, "INCREMENT", 500)]
    batch, n = select(pool, 10, 4, 20)
    assert list(batch) == [0, 1], f"INCREMENT requests always run; expected [0, 1], got {list(batch)}"
    assert n == 10, f"INCREMENT requests must not change n_rsrv; expected 10, got {n}"


def test_running_requests_are_skipped():
    pool = [req(0, "RUNNING", 10), req(1, "INCREMENT", 10)]
    batch, _ = select(pool, 0, 4, 100)
    assert list(batch) == [1], f"RUNNING requests are already in the engine; expected [1], got {list(batch)}"


def test_max_batch_size():
    pool = [req(i, "INCREMENT", 1) for i in range(5)]
    batch, _ = select(pool, 0, 3, 100)
    assert list(batch) == [0, 1, 2], f"expected the 3 earliest requests [0, 1, 2], got {list(batch)}"


def test_arrival_order_not_pool_order():
    pool = [req(5, "INCREMENT", 1), req(1, "INCREMENT", 1), req(3, "INCREMENT", 1)]
    batch, _ = select(pool, 0, 2, 100)
    assert list(batch) == [1, 2], f"the two earliest arrivals are pool indices 1 and 2; got {list(batch)}"


def test_reservation_accumulates_across_requests():
    pool = [req(0, "INITIATION", 30), req(1, "INITIATION", 30), req(2, "INITIATION", 30), req(3, "INITIATION", 30)]
    batch, n = select(pool, 0, 8, 100)
    assert list(batch) == [0, 1, 2] and n == 90, f"3 of 4 requests fit in 100 slots; got batch {list(batch)}, n_rsrv {n}"
