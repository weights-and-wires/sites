from submission import attention_fraction, block_params, n_params, petaflop_days, train_flops


def test_block_params():
    got = block_params(768)
    assert got == 7087872, f"expected 12*768**2 + 13*768 = 7087872, got {got}"
    assert block_params(12288) == 1812099072, f"expected 1812099072 for d=12288, got {block_params(12288)}"


def test_small_model():
    got = n_params(12, 768)
    assert got == 125226240, f"expected 125226240 for GPT-3 Small, got {got}"


def test_175b_matches_table_d1():
    got = n_params(96, 12288)
    assert got == 174604259328, f"expected 174604259328 for GPT-3 175B, got {got}"
    assert abs(got / 174.6e9 - 1) < 1e-3, f"expected within 0.1% of Table D.1's 174.6B, got {got}"


def test_embeddings_are_counted_once():
    d = 1024
    delta = n_params(24, d, n_vocab=50258) - n_params(24, d)
    assert delta == d, f"one extra vocabulary row adds d = {d} parameters when the output projection is shared; got {delta}"


def test_positions_are_counted():
    d = 1024
    delta = n_params(24, d, n_ctx=2049) - n_params(24, d)
    assert delta == d, f"one extra position adds d = {d} parameters; got {delta}"


def test_layers_add_whole_blocks():
    d = 512
    delta = n_params(7, d) - n_params(6, d)
    assert delta == block_params(d), f"one extra layer adds {block_params(d)} parameters, got {delta}"


def test_train_flops():
    got = train_flops(174.6e9, 300e9)
    assert abs(got / 3.1428e23 - 1) < 1e-9, f"expected 3.1428e23, got {got:.4e}"
    assert train_flops(1, 1) == 6, f"expected 6 flops for one parameter and one token, got {train_flops(1, 1)}"


def test_petaflop_days():
    got = petaflop_days(3.14e23)
    assert abs(got - 3634.26) < 0.01, f"expected about 3634.26 PF-days for 3.14e23 flops, got {got}"
    assert abs(petaflop_days(8.64e19) - 1.0) < 1e-12, f"expected 1 PF-day for 8.64e19 flops, got {petaflop_days(8.64e19)}"


def test_attention_fraction():
    got = attention_fraction(12288, 2048)
    assert abs(got - 0.027030) < 1e-5, f"expected 0.02703 for the 175B model, got {got}"
    small = attention_fraction(768, 2048)
    assert abs(small - 0.307692) < 1e-5, f"expected 0.30769 for GPT-3 Small, got {small}"
