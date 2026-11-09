from submission import kv_cache_bytes, mamba_state_bytes, crossover_length


def test_kv_cache_100k():
    # Table 12, 1.3B Transformer: 24 layers, 32 heads of width 64, fp16.
    got = kv_cache_bytes(24, 32, 64, 100_000)
    assert got == 19_660_800_000, f"expected 19660800000, got {got}"


def test_kv_cache_is_linear_in_length_and_dtype():
    one = kv_cache_bytes(2, 4, 8, 1)
    assert one == 2 * 2 * 4 * 8 * 2, f"one token, fp16: expected {2 * 2 * 4 * 8 * 2}, got {one}"
    assert kv_cache_bytes(2, 4, 8, 50) == 50 * one, "the cache must grow linearly with seq_len"
    assert kv_cache_bytes(2, 4, 8, 1, bytes_per_elem=4) == 2 * one, "fp32 must double the fp16 size"


def test_mamba_state_value():
    got = mamba_state_bytes(48, 2048)
    assert got == 7_471_104, f"expected 7471104 (48 layers, d_inner 4096, N 16, conv width 4, fp16), got {got}"


def test_mamba_state_parts():
    # One layer, d_model 1, expand 2: d_inner = 2.
    ssm_only = mamba_state_bytes(1, 1, expand=2, d_state=5, d_conv=1, bytes_per_elem=1)
    assert ssm_only == 10, f"with d_conv=1 only h remains: expected 2 * 5 = 10, got {ssm_only}"
    both = mamba_state_bytes(1, 1, expand=2, d_state=5, d_conv=4, bytes_per_elem=1)
    assert both == 10 + 2 * 3, f"the conv buffer adds d_inner * (d_conv - 1) = 6: expected 16, got {both}"
    assert mamba_state_bytes(3, 1, expand=2, d_state=5, d_conv=4, bytes_per_elem=1) == 48, "the state scales with layers"


def test_crossover_ties_do_not_count():
    state = mamba_state_bytes(48, 2048)
    n = crossover_length(24, 32, 64, state)
    assert n == 39, f"the cache equals the state at 38 tokens, so strictly larger starts at 39; got {n}"


def test_crossover_matches_brute_force():
    for state in [1, 100, 1000, 4096, 12_345]:
        n = crossover_length(2, 4, 8, state)
        assert kv_cache_bytes(2, 4, 8, n) > state, f"state={state}: the cache at {n} tokens must exceed the state"
        assert kv_cache_bytes(2, 4, 8, n - 1) <= state, f"state={state}: the cache at {n - 1} tokens must not exceed the state"
