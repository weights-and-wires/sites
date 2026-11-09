from submission import latent_shape, compression_ratio, position_ratio, attention_ratio


def test_latent_shape_square():
    got = tuple(latent_shape(512, 512, 8, 4))
    assert got == (64, 64, 4), f"expected (64, 64, 4), got {got}"


def test_latent_shape_rectangular():
    got = tuple(latent_shape(512, 768, 8, 4))
    assert got == (64, 96, 4), f"expected (64, 96, 4) for a 512x768 image, got {got}"


def test_latent_shape_rejects_bad_factor():
    try:
        latent_shape(500, 500, 8, 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError because 8 does not divide 500, but nothing was raised")


def test_compression_ratio_paper_settings():
    # (f, c) pairs from Table 13: 512x512x3 = 786,432 values.
    want = {(4, 3): 16.0, (8, 4): 48.0, (16, 8): 96.0, (32, 32): 96.0}
    for (f, c), r in want.items():
        got = compression_ratio(512, 512, f, c)
        assert abs(got - r) < 1e-9, f"f={f}, c={c}: expected {r}, got {got}"


def test_compression_ratio_depends_on_channels():
    a, b = compression_ratio(256, 256, 8, 4), compression_ratio(256, 256, 8, 16)
    assert abs(a - 48.0) < 1e-9 and abs(b - 12.0) < 1e-9, f"expected 48.0 and 12.0, got {a} and {b}"


def test_position_ratio():
    got = position_ratio(512, 768, 8)
    assert abs(got - 64.0) < 1e-9, f"expected 64.0 (that is f squared), got {got}"


def test_attention_ratio():
    got = attention_ratio(512, 512, 8)
    assert abs(got - 4096.0) < 1e-6, f"expected 4096.0 (that is f to the 4th power), got {got}"
    got = attention_ratio(512, 768, 4)
    assert abs(got - 256.0) < 1e-6, f"expected 256.0, got {got}"
