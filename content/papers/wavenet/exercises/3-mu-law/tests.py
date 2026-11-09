import numpy as np
from submission import mu_law_encode, mu_law_decode


def test_table_values():
    # Values from the page (scripts/mu_law_example.py).
    xs = np.array([0.001, 0.01, 0.1, 0.5, 0.9])
    want = np.array([133, 157, 203, 239, 253])
    got = mu_law_encode(xs)
    assert np.array_equal(got, want), f"expected {want}, got {got}"


def test_endpoints_and_zero():
    got = mu_law_encode(np.array([-1.0, 0.0, 1.0]))
    assert np.array_equal(got, [0, 128, 255]), f"expected [0, 128, 255], got {got}"


def test_integer_range():
    x = np.linspace(-1, 1, 1001)
    q = mu_law_encode(x)
    assert np.issubdtype(q.dtype, np.integer), f"expected an integer dtype, got {q.dtype}"
    assert q.min() == 0 and q.max() == 255, f"expected range 0..255, got {q.min()}..{q.max()}"


def test_monotonic():
    q = mu_law_encode(np.linspace(-1, 1, 200001))
    assert np.all(np.diff(q) >= 0), "encoding must not decrease as x increases"
    assert len(np.unique(q)) == 256, f"expected all 256 levels used, got {len(np.unique(q))}"


def test_odd_symmetry_of_decode():
    q = np.arange(0, 256)
    d = mu_law_decode(q)
    # q and 255 - q are mirror images, so decoded values are negatives.
    assert np.allclose(d, -d[::-1], atol=1e-12), "decode(q) must equal -decode(255 - q)"


def test_decode_endpoints():
    got = mu_law_decode(np.array([0, 255]))
    assert np.allclose(got, [-1.0, 1.0]), f"expected [-1, 1], got {got}"


def test_round_trip_error():
    x = np.linspace(-0.999, 0.999, 20001)
    err = np.abs(mu_law_decode(mu_law_encode(x)) - x)
    assert err.max() < 0.0216, f"expected max round-trip error below 0.0216, got {err.max():.5f}"
    small = np.abs(x) < 0.01
    assert err[small].max() < 0.0004, (
        f"quiet samples (|x| < 0.01) should have error below 0.0004, got {err[small].max():.5f}"
    )


def test_quiet_resolution_beats_linear():
    # Linear 8-bit has a worst-case error of 1/255 = 0.00392 everywhere.
    x = np.linspace(-0.01, 0.01, 2001)
    err = np.abs(mu_law_decode(mu_law_encode(x)) - x).max()
    assert err < 0.001, f"expected error well below linear 8-bit (0.0039), got {err:.5f}"


def test_encode_decode_is_identity_on_codes():
    q = np.arange(256)
    got = mu_law_encode(mu_law_decode(q))
    assert np.array_equal(got, q), f"expected codes to round-trip, first mismatch at {np.nonzero(got != q)[0][:5]}"


def test_other_mu():
    got = mu_law_encode(np.array([-1.0, 0.0, 1.0]), mu=15)
    assert np.array_equal(got, [0, 8, 15]), f"mu=15: expected [0, 8, 15], got {got}"
