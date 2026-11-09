from submission import non_embedding_params, embedding_params, forward_flops_per_token, training_compute_pf_days


def test_standard_shape_is_12_l_d_squared():
    got = non_embedding_params(12, 768)
    assert got == 12 * 12 * 768**2, f"expected {12 * 12 * 768**2}, got {got}"
    got = non_embedding_params(48, 1600)
    assert got == 1_474_560_000, f"expected 1474560000, got {got}"


def test_nonstandard_feed_forward_width():
    got = non_embedding_params(6, 512, d_ff=2048)
    assert got == 18_874_368, f"expected 18874368 with d_ff = 2048, got {got}"
    got = non_embedding_params(6, 512, d_ff=1024)
    assert got == 12_582_912, f"expected 12582912 with d_ff = 1024, got {got}"


def test_nonstandard_attention_width():
    got = non_embedding_params(4, 256, d_attn=512)
    want = 2 * 256 * 4 * (2 * 512 + 1024)
    assert got == want, f"expected {want} with d_attn = 512, got {got}"


def test_embedding_params():
    got = embedding_params(50257, 1024, 1600)
    assert got == 82_049_600, f"expected 82049600, got {got}"
    assert embedding_params(50257, 1024, 768) == 39_383_808, f"expected 39383808, got {embedding_params(50257, 1024, 768)}"


def test_embeddings_are_not_in_n():
    n = non_embedding_params(2, 128)
    assert n == 2 * 128 * 2 * (2 * 128 + 512), f"expected {2 * 128 * 2 * (2 * 128 + 512)}, got {n}"


def test_forward_flops_includes_context_term():
    got = forward_flops_per_token(48, 1600, 1024)
    want = 2 * 1_474_560_000 + 2 * 48 * 1024 * 1600
    assert got == want, f"expected {want}, got {got}"


def test_forward_context_term_uses_d_attn():
    got = forward_flops_per_token(4, 256, 1024, d_attn=512)
    n = 2 * 256 * 4 * (2 * 512 + 1024)
    want = 2 * n + 2 * 4 * 1024 * 512
    assert got == want, f"expected {want} (context term uses d_attn), got {got}"


def test_training_compute_in_pf_days():
    got = training_compute_pf_days(1_474_560_000, 512 * 1024, 250_000)
    assert abs(got - 13.42) < 0.01, f"expected about 13.42 PF-days, got {got}"


def test_pf_day_is_8_64e19_flops():
    got = training_compute_pf_days(1e10, 1e3, 1.0)  # 6e13 FLOPs
    want = 6e13 / 8.64e19
    assert abs(got - want) / want < 1e-9, f"expected {want}, got {got}"


def test_compute_is_linear_in_each_factor():
    base = training_compute_pf_days(1e8, 1e5, 1e3)
    assert abs(training_compute_pf_days(2e8, 1e5, 1e3) / base - 2) < 1e-9, "doubling N must double C"
    assert abs(training_compute_pf_days(1e8, 2e5, 1e3) / base - 2) < 1e-9, "doubling B must double C"
    assert abs(training_compute_pf_days(1e8, 1e5, 2e3) / base - 2) < 1e-9, "doubling S must double C"
