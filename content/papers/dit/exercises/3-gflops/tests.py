from submission import dit_gflops

CONFIGS = {"S": (12, 384), "B": (12, 768), "L": (24, 1024), "XL": (28, 1152)}
# Table 4: Gflops at I = 32.
TABLE4 = {
    ("S", 8): 0.36, ("S", 4): 1.41, ("S", 2): 6.06,
    ("B", 8): 1.42, ("B", 4): 5.56, ("B", 2): 23.01,
    ("L", 8): 5.01, ("L", 4): 19.70, ("L", 2): 80.71,
    ("XL", 8): 7.39, ("XL", 4): 29.05, ("XL", 2): 118.64,
}


def test_matches_table_4():
    for (name, p), want in TABLE4.items():
        N, d = CONFIGS[name]
        got = dit_gflops(N, d, p)
        assert abs(got - want) / want < 0.01, f"DiT-{name}/{p}: expected {want} within 1%, got {got:.3f}"


def test_xl2_at_512():
    got = dit_gflops(28, 1152, 2, I=64)
    assert abs(got - 524.6) / 524.6 < 0.01, f"DiT-XL/2 on a 64x64 latent: expected 524.6 within 1%, got {got:.2f}"


def test_halving_patch_size_roughly_quadruples_flops():
    for name, (N, d) in CONFIGS.items():
        r = dit_gflops(N, d, 2) / dit_gflops(N, d, 4)
        assert 4.0 < r < 4.4, f"DiT-{name}: /2 over /4 should be a bit above 4 (attention grows as T^2), got {r:.3f}"


def test_attention_term_grows_quadratically_in_tokens():
    # With d tiny, the T^2 d attention term is small, but the ratio must still exceed the pure T ratio.
    N, d = 12, 384
    big = dit_gflops(N, d, 1, I=32)
    small = dit_gflops(N, d, 2, I=32)
    assert big / small > 4.0, f"quartering the patch area must more than quadruple the cost; ratio {big / small:.3f}"


def test_cost_scales_with_depth():
    a = dit_gflops(12, 384, 4)
    b = dit_gflops(24, 384, 4)
    assert 1.9 < b / a < 2.0, f"doubling N should nearly double Gflops (patchify and decoder do not scale); got ratio {b / a:.3f}"
