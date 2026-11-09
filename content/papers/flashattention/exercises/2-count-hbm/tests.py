from submission import block_sizes, standard_hbm_accesses, flash_hbm_accesses


def test_block_sizes():
    assert block_sizes(64, 65536) == (256, 64), f"expected (256, 64), got {block_sizes(64, 65536)}"
    assert block_sizes(64, 1000) == (4, 4), f"Bc = ceil(1000/256) = 4 and Br = min(4, 64) = 4; got {block_sizes(64, 1000)}"
    assert block_sizes(2, 32) == (4, 2), f"expected (4, 2), got {block_sizes(2, 32)}"


def test_standard_small():
    assert standard_hbm_accesses(4, 2) == 96, f"N=4, d=2: 4*N*d + 4*N^2 = 96, got {standard_hbm_accesses(4, 2)}"


def test_standard_gpt2_size():
    got = standard_hbm_accesses(1024, 64)
    assert got == 4_456_448, f"N=1024, d=64: expected 4456448, got {got}"


def test_standard_is_quadratic():
    a, b = standard_hbm_accesses(4096, 64), standard_hbm_accesses(8192, 64)
    assert 3.9 < b / a < 4.1, f"doubling N should quadruple the traffic when N >> d; ratio was {b / a:.3f}"


def test_flash_small():
    assert flash_hbm_accesses(8, 2, 32) == 192, f"N=8, d=2, M=32: Bc=4, Tc=2; expected 32 + 2*(48+32) = 192, got {flash_hbm_accesses(8, 2, 32)}"


def test_flash_gpt2_size():
    got = flash_hbm_accesses(1024, 64, 65536)
    assert got == 933_888, f"N=1024, d=64, M=65536: expected 933888, got {got}"


def test_partial_last_block_costs_a_full_pass():
    # N=10, d=2, M=16: Bc=2, Tc=5 exactly. M=24: Bc=3, Tc=ceil(10/3)=4 (not 3).
    got = flash_hbm_accesses(10, 2, 24)
    want = 2 * 10 * 2 + 4 * (2 * 10 * 2 + 2 * 10 + 10 * 2 + 2 * 10)
    assert got == want, f"Tc must round up to 4; expected {want}, got {got}"


def test_more_sram_means_less_traffic():
    a = flash_hbm_accesses(1024, 64, 32768)
    b = flash_hbm_accesses(1024, 64, 65536)
    c = flash_hbm_accesses(1024, 64, 131072)
    assert a > b > c, f"traffic must fall as M grows; got {a}, {b}, {c}"
    assert 0.5 < c / b < 0.6, f"doubling M should nearly halve the traffic; ratio was {c / b:.3f}"


def test_flash_beats_standard_by_about_M_over_3d2():
    N, d, M = 65536, 64, 65536
    ratio = standard_hbm_accesses(N, d) / flash_hbm_accesses(N, d, M)
    want = M / (3 * d * d)
    assert abs(ratio - want) / want < 0.05, f"expected the ratio near M/(3 d^2) = {want:.2f}, got {ratio:.2f}"
