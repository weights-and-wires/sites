from submission import kv_bytes_per_token, min_bandwidth_gbit, transfer_seconds


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(b))


def test_opt_66b_bytes_per_token():
    got = kv_bytes_per_token(64, 72, 128)
    assert got == 2359296, f"2 * 64 * 72 * 128 * 2 = 2359296, got {got}"


def test_grouped_query_attention_shrinks_the_cache():
    got = kv_bytes_per_token(80, 8, 128)
    assert got == 327680, f"2 * 80 * 8 * 128 * 2 = 327680, got {got}"


def test_bytes_per_value_is_a_parameter():
    got = kv_bytes_per_token(64, 72, 128, bytes_per_value=1)
    assert got == 1179648, f"an fp8 cache is half of fp16: 1179648, got {got}"
    got = kv_bytes_per_token(1, 1, 1)
    assert got == 4, f"one key and one value in fp16 is 4 bytes, got {got}"


def test_transfer_seconds_uses_bits():
    got = transfer_seconds(1_250_000_000, 100)
    assert close(got, 0.1), f"1.25 GB over 100 Gbit/s is 0.1 s, got {got}"
    got = transfer_seconds(1_207_959_552, 25)
    assert close(got, 0.38654705664), f"512 tokens of OPT-66B over 25 Gbit/s, expected about 0.38655 s, got {got}"


def test_transfer_scales_inversely_with_bandwidth():
    slow = transfer_seconds(10**9, 10)
    fast = transfer_seconds(10**9, 400)
    assert close(slow / fast, 40), f"40 times the bandwidth is 40 times faster, got ratio {slow / fast}"


def test_min_bandwidth():
    got = min_bandwidth_gbit(1_207_959_552, 10)
    assert close(got, 96.63676416), f"expected about 96.64 Gbit/s, got {got}"
    got = min_bandwidth_gbit(125_000_000, 1)
    assert close(got, 1.0), f"125 MB per second is 1 Gbit/s, got {got}"


def test_min_bandwidth_is_linear_in_rate():
    a = min_bandwidth_gbit(10**9, 3)
    b = min_bandwidth_gbit(10**9, 6)
    assert close(b, 2 * a), f"doubling the rate doubles the bandwidth: {a} -> {b}"


def test_transfer_time_and_bandwidth_agree():
    n = 10**9
    rate = 5
    bw = min_bandwidth_gbit(n, rate)
    got = transfer_seconds(n, bw)
    assert close(got, 1 / rate), f"a link sized for {rate} requests/s moves one in {1 / rate} s, got {got}"
