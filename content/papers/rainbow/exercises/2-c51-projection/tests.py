import numpy as np
from submission import project_distribution

z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])


def test_page_example():
    got = np.asarray(project_distribution(p, 0.5, 0.9, z))
    want = np.array([0.03, 0.15, 0.32, 0.32, 0.18])
    assert got.shape == (5,), f"expected shape (5,), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_probability_is_conserved():
    for r, g in [(0.5, 0.9), (-1.3, 0.7), (3.0, 0.9), (0.0, 0.0), (-0.25, 1.0)]:
        got = np.asarray(project_distribution(p, r, g, z))
        assert np.isclose(got.sum(), 1.0), f"r={r}, gamma_n={g}: mass should sum to 1, got {got.sum()} from {got}"
        assert (got >= -1e-12).all(), f"r={r}, gamma_n={g}: negative mass in {got}"


def test_identity_when_nothing_moves():
    got = np.asarray(project_distribution(p, 0.0, 1.0, z))
    assert np.allclose(got, p), f"r = 0 and gamma_n = 1 must leave p unchanged, expected {p}, got {got}"


def test_shift_by_one_atom():
    got = np.asarray(project_distribution(p, 1.0, 1.0, z))
    want = np.array([0.0, 0.1, 0.2, 0.4, 0.3])
    assert np.allclose(got, want), f"a shift by one atom with the top clipped, expected {want}, got {got}"


def test_terminal_collapses_onto_the_reward():
    got = np.asarray(project_distribution(p, 0.25, 0.0, z))
    want = np.array([0.0, 0.0, 0.75, 0.25, 0.0])
    assert np.allclose(got, want), f"gamma_n = 0 and r = 0.25 put all mass between atoms 0 and 1: expected {want}, got {got}"


def test_clipping_at_both_ends():
    got = np.asarray(project_distribution(p, 5.0, 0.9, z))
    want = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(got, want), f"every atom is above v_max, expected {want}, got {got}"
    got = np.asarray(project_distribution(p, -5.0, 0.9, z))
    want = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(got, want), f"every atom is below v_min, expected {want}, got {got}"


def test_mean_is_kept_away_from_the_edges():
    q = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
    got = np.asarray(project_distribution(q, 0.3, 0.5, z))
    assert np.isclose(z @ got, 0.3 + 0.5 * (z @ q)), f"mean should be 0.3, got {z @ got}"


def test_other_grid_spacing():
    zz = np.linspace(-10, 10, 51)
    q = np.zeros(51)
    q[25] = 1.0
    got = np.asarray(project_distribution(q, 1.0, 0.97, zz))
    assert np.isclose(got[27], 0.5) and np.isclose(got[28], 0.5), f"1.0 sits halfway between atoms 27 and 28 (spacing 0.4 puts 1.0 at index 27.5), got {got[25:30]}"
