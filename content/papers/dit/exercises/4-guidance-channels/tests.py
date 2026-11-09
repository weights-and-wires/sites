import numpy as np
from submission import guided_output

rng = np.random.default_rng(0)
OC = rng.normal(size=(8, 2, 2))
OU = rng.normal(size=(8, 2, 2))


def test_shape_and_input_untouched():
    oc, ou = OC.copy(), OU.copy()
    got = guided_output(oc, ou, 1.5)
    assert got.shape == (8, 2, 2), f"expected shape (8, 2, 2), got {got.shape}"
    assert np.array_equal(oc, OC), "out_cond must not be modified in place"
    assert np.array_equal(ou, OU), "out_uncond must not be modified in place"


def test_scale_one_returns_conditional():
    got = guided_output(OC, OU, 1.0)
    assert np.allclose(got, OC), f"s = 1 must recover the conditional output; max error {np.abs(got - OC).max()}"


def test_formula_on_guided_channels():
    s = 4.0
    got = guided_output(OC, OU, s)
    want = OU[:3] + s * (OC[:3] - OU[:3])
    assert np.allclose(got[:3], want), f"expected {want[0, 0]}, got {got[0, 0]}"


def test_fourth_noise_channel_is_not_guided():
    got = guided_output(OC, OU, 4.0)
    assert np.allclose(got[3], OC[3]), f"channel 3 must copy the conditional pass; expected {OC[3, 0]}, got {got[3, 0]}"


def test_covariance_channels_are_not_guided():
    got = guided_output(OC, OU, 4.0)
    assert np.allclose(got[4:], OC[4:]), f"covariance channels must copy the conditional pass; expected {OC[4, 0]}, got {got[4, 0]}"


def test_page_example():
    oc = np.arange(1, 9, dtype=float).reshape(8, 1, 1)
    ou = oc * 0.5
    got = guided_output(oc, ou, 1.5)[:, 0, 0]
    want = [1.25, 2.5, 3.75, 4, 5, 6, 7, 8]
    assert np.allclose(got, want), f"expected {want}, got {got.tolist()}"


def test_n_guided_argument():
    got = guided_output(OC, OU, 2.0, n_guided=4)
    want = OU[:4] + 2.0 * (OC[:4] - OU[:4])
    assert np.allclose(got[:4], want), "n_guided=4 must guide all four noise channels"
    assert np.allclose(got[4:], OC[4:]), "covariance channels still copy the conditional pass"
