from submission import model_state_bytes, max_params

GB = 1e9


def test_ddp_is_16_bytes_per_param():
    got = model_state_bytes(1_000, 8, 0)
    assert got == 16_000, f"expected 16000 bytes (2 + 2 + 12 per parameter), got {got}"


def test_ddp_ignores_worker_count():
    a, b = model_state_bytes(1e9, 1, 0), model_state_bytes(1e9, 1024, 0)
    assert a == b, f"data parallelism replicates everything, so expected equal values, got {a} and {b}"


def test_table1_stage1():
    got = model_state_bytes(7.5e9, 64, 1) / GB
    assert abs(got - 31.4) < 0.05, f"Figure 1 says 31.4 GB for 7.5B at Nd=64 with stage 1, got {got:.2f}"


def test_table1_stage2():
    got = model_state_bytes(7.5e9, 64, 2) / GB
    assert abs(got - 16.6) < 0.05, f"Figure 1 says 16.6 GB for 7.5B at Nd=64 with stage 2, got {got:.2f}"


def test_table1_stage3():
    got = model_state_bytes(7.5e9, 64, 3) / GB
    assert abs(got - 1.88) < 0.01, f"Figure 1 says 1.9 GB for 7.5B at Nd=64 with stage 3, got {got:.3f}"


def test_table1_other_rows():
    cases = [(128e9, 4, 1, 896), (128e9, 4, 2, 704), (128e9, 16, 3, 128), (1e12, 1024, 2, 2013), (1e12, 256, 1, 4046)]
    for psi, nd, stage, want in cases:
        got = model_state_bytes(psi, nd, stage) / GB
        assert abs(got - want) < 1, f"psi={psi:.0e}, nd={nd}, stage={stage}: expected {want} GB, got {got:.1f}"


def test_large_nd_limits():
    psi = 1e9
    assert abs(model_state_bytes(psi, 1e9, 1) / psi - 4) < 1e-6, "stage 1 should approach 4 bytes per parameter (4x less than 16)"
    assert abs(model_state_bytes(psi, 1e9, 2) / psi - 2) < 1e-6, "stage 2 should approach 2 bytes per parameter (8x less than 16)"
    assert model_state_bytes(psi, 1e9, 3) / psi < 1e-6, "stage 3 should approach 0 bytes per parameter"


def test_k_is_used():
    got = model_state_bytes(1000, 4, 1, K=8)
    want = 2000 + 2000 + 8000 / 4
    assert got == want, f"with K=8 and Nd=4, expected {want}, got {got}"


def test_max_params_section_5_4():
    for stage, want in [(0, 2.0e9), (1, 7.6e9), (2, 14.4e9), (3, 128e9)]:
        got = max_params(32 * GB, 64, stage)
        assert abs(got - want) / want < 0.01, f"stage {stage}: expected about {want:.3g} parameters in 32 GB, got {got:.3g}"


def test_max_params_roundtrip():
    psi = max_params(32 * GB, 16, 2)
    got = model_state_bytes(psi, 16, 2)
    assert abs(got - 32 * GB) < 1, f"model states for max_params should fill 32 GB, got {got}"
