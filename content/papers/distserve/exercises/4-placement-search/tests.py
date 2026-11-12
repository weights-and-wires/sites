from submission import best_split, replicas_needed


def close(a, b):
    return abs(a - b) < 1e-9


def test_replicas_round_up():
    got = replicas_needed(10.0, 3.0)
    assert got == 4, f"10 / 3 rounds up to 4, got {got}"
    got = replicas_needed(10.0, 20.0)
    assert got == 1, f"one instance is enough, got {got}"


def test_replicas_exact_multiple():
    got = replicas_needed(0.3, 0.1)
    assert got == 3, f"0.3 / 0.1 is 3 up to floating-point error, expected 3, got {got}"
    got = replicas_needed(2.1, 0.3)
    assert got == 7, f"2.1 / 0.3 is 7 up to floating-point error, expected 7, got {got}"
    got = replicas_needed(9.0, 3.0)
    assert got == 3, f"expected 3, got {got}"


def test_budget_three():
    rate, n_p, n_d = best_split(3, 6.8, 17.2)
    assert (n_p, n_d) == (2, 1), f"2 prefill + 1 decoding gives min(13.6, 17.2), got {(n_p, n_d)}"
    assert close(rate, 13.6), f"expected rate 13.6, got {rate}"


def test_budget_four_decode_is_the_limit():
    rate, n_p, n_d = best_split(4, 6.8, 17.2)
    assert (n_p, n_d) == (3, 1), f"3 prefill + 1 decoding gives min(20.4, 17.2) = 17.2, got {(n_p, n_d)}"
    assert close(rate, 17.2), f"expected rate 17.2, got {rate}"


def test_budget_eight_uses_all_gpus():
    rate, n_p, n_d = best_split(8, 6.8, 17.2)
    assert (n_p, n_d) == (6, 2), f"6 prefill + 2 decoding gives min(40.8, 34.4) = 34.4, got {(n_p, n_d)}"
    assert close(rate, 34.4), f"expected rate 34.4, got {rate}"


def test_every_phase_gets_a_gpu():
    rate, n_p, n_d = best_split(5, 1.0, 100.0)
    assert (n_p, n_d) == (4, 1), f"decoding is so fast that 1 GPU is enough: 4 prefill + 1 decoding, got {(n_p, n_d)}"
    assert close(rate, 4.0), f"expected rate 4.0, got {rate}"
    rate, n_p, n_d = best_split(5, 100.0, 1.0)
    assert (n_p, n_d) == (1, 4), f"expected 1 prefill + 4 decoding, got {(n_p, n_d)}"


def test_ties_use_fewer_gpus():
    rate, n_p, n_d = best_split(5, 1.0, 1.0)
    assert close(rate, 2.0), f"expected rate 2.0, got {rate}"
    assert (n_p, n_d) == (2, 2), f"2 + 2 reaches 2.0 on 4 GPUs, so the fifth GPU is not used, got {(n_p, n_d)}"
    rate, n_p, n_d = best_split(3, 1.0, 5.0)
    assert (rate, n_p, n_d) == (2.0, 2, 1), f"expected (2.0, 2, 1), got {(rate, n_p, n_d)}"


def test_too_small_budget():
    got = best_split(1, 1.0, 1.0)
    assert got == (0.0, 0, 0), f"one GPU cannot hold both phases, expected (0.0, 0, 0), got {got}"
    got = best_split(2, 2.0, 3.0)
    assert got == (2.0, 1, 1), f"two GPUs: one of each, rate min(2, 3) = 2, got {got}"
