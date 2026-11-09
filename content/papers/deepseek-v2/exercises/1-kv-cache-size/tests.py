from submission import cache_mha, cache_gqa, cache_mla, equivalent_gqa_groups

N_H, D_H, L, D_C, D_HR = 128, 128, 60, 512, 64


def test_mha():
    got = cache_mha(N_H, D_H, L)
    assert got == 1_966_080, f"expected 1966080, got {got}"


def test_mha_small():
    got = cache_mha(2, 3, 4)
    assert got == 48, f"expected 2*2*3*4 = 48, got {got}"


def test_gqa():
    got = cache_gqa(8, D_H, L)
    assert got == 122_880, f"expected 122880, got {got}"


def test_gqa_extremes():
    assert cache_gqa(N_H, D_H, L) == cache_mha(N_H, D_H, L), "n_g = n_h must equal MHA"
    assert cache_gqa(1, D_H, L) == 15_360, f"n_g = 1 is MQA; expected 15360, got {cache_gqa(1, D_H, L)}"


def test_mla():
    got = cache_mla(D_C, D_HR, L)
    assert got == 34_560, f"expected (512 + 64) * 60 = 34560, got {got}"


def test_mla_counts_rope_key_and_no_values():
    got = cache_mla(10, 3, 2)
    assert got == 26, f"expected (10 + 3) * 2 = 26, got {got}"


def test_equivalent_groups():
    got = equivalent_gqa_groups(D_C, D_HR, D_H)
    assert abs(got - 2.25) < 1e-9, f"expected 2.25 (Table 1), got {got}"


def test_equivalent_groups_matches_cache():
    n_g = equivalent_gqa_groups(D_C, D_HR, D_H)
    assert abs(cache_gqa(n_g, D_H, L) - cache_mla(D_C, D_HR, L)) < 1e-6, "GQA at that n_g must cache as much as MLA"
