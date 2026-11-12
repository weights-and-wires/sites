import numpy as np
from submission import add_root_noise

P = np.array([0.7, 0.2, 0.1, 0.0])
LEGAL = np.array([True, True, True, False])


def test_matches_the_reference_draw():
    got = add_root_noise(P, LEGAL, 0.25, 0.3, np.random.default_rng(5))
    eta = np.zeros(4)
    eta[:3] = np.random.default_rng(5).dirichlet(np.full(3, 0.3))
    want = 0.75 * P + 0.25 * eta
    assert np.allclose(got, want, atol=1e-12), f"expected {want}, got {got}"


def test_result_is_a_distribution_with_no_mass_on_illegal_moves():
    rng = np.random.default_rng(0)
    for _ in range(20):
        got = add_root_noise(P, LEGAL, 0.25, 0.3, rng)
        assert abs(got.sum() - 1.0) < 1e-9, f"expected the mixed prior to sum to 1, got {got.sum()} for {got}"
        assert got[3] == 0.0, f"expected 0 on the illegal move, got {got}"
        assert np.all(got >= 0.75 * P - 1e-12), f"every legal prior keeps at least (1 - eps) p, got {got}"


def test_eps_zero_keeps_the_prior_and_eps_one_replaces_it():
    got = add_root_noise(P, LEGAL, 0.0, 0.3, np.random.default_rng(1))
    assert np.allclose(got, P), f"expected eps = 0 to return p = {P}, got {got}"
    got = add_root_noise(P, LEGAL, 1.0, 0.3, np.random.default_rng(1))
    assert abs(got.sum() - 1.0) < 1e-9 and not np.allclose(got, P), f"expected eps = 1 to return pure noise, got {got}"


def test_mean_is_the_mix_of_prior_and_uniform():
    rng = np.random.default_rng(2)
    draws = np.array([add_root_noise(P, LEGAL, 0.25, 0.3, rng) for _ in range(4000)])
    want = 0.75 * P + 0.25 * np.array([1 / 3, 1 / 3, 1 / 3, 0.0])
    assert np.allclose(draws.mean(axis=0), want, atol=0.01), f"expected mean {want}, got {np.round(draws.mean(axis=0), 4)}"


def test_small_alpha_gives_spiky_noise_and_large_alpha_flat_noise():
    p = np.full(30, 1 / 30)
    legal = np.ones(30, dtype=bool)
    rng = np.random.default_rng(4)

    def top_noise(alpha):
        tops = []
        for _ in range(200):
            noise = (add_root_noise(p, legal, 0.25, alpha, rng) - 0.75 * p) / 0.25
            tops.append(noise.max())
        return float(np.mean(tops))

    spiky, flat = top_noise(0.03), top_noise(10.0)
    assert spiky > 0.5, f"expected alpha = 0.03 to put over half the noise on one move, got {spiky:.3f}"
    assert flat < 0.15, f"expected alpha = 10 to spread the noise out (largest share < 0.15), got {flat:.3f}"


def test_inputs_are_not_modified():
    p, legal = P.copy(), LEGAL.copy()
    add_root_noise(p, legal, 0.25, 0.3, np.random.default_rng(6))
    assert np.array_equal(p, P), f"expected p unchanged, got {p}"
    assert np.array_equal(legal, LEGAL), f"expected legal unchanged, got {legal}"
