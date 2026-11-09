import numpy as np
from submission import preference_loss, preference_prob


def test_equation_1_value():
    got = float(preference_prob(3.0, 1.0))
    assert abs(got - 0.8808) < 1e-4, f"expected 0.8808 for returns 3 and 1, got {got}"


def test_equal_returns_is_a_coin_flip():
    got = float(preference_prob(5.0, 5.0))
    assert abs(got - 0.5) < 1e-12, f"expected 0.5 for equal returns, got {got}"


def test_only_the_difference_matters():
    a = preference_prob(np.array([3.0, 0.5]), np.array([1.0, 2.0]))
    b = preference_prob(np.array([103.0, 100.5]), np.array([101.0, 102.0]))
    assert np.allclose(a, b), f"shifting both returns must not change P; expected {a}, got {b}"


def test_probabilities_of_both_orders_sum_to_one():
    r1, r2 = np.array([0.3, -2.0, 7.0]), np.array([1.1, 0.4, 6.5])
    for eps in (0.0, 0.1):
        total = preference_prob(r1, r2, eps) + preference_prob(r2, r1, eps)
        assert np.allclose(total, 1.0), f"eps={eps}: expected P12 + P21 = 1, got {total}"


def test_eps_adjustment():
    got = float(preference_prob(3.0, 1.0, 0.1))
    assert abs(got - 0.8427) < 1e-4, f"expected 0.9 * 0.8808 + 0.05 = 0.8427, got {got}"
    far = float(preference_prob(1000.0, 0.0, 0.1))
    assert abs(far - 0.95) < 1e-9, f"for a huge gap expected 1 - eps/2 = 0.95, got {far}"


def test_loss_values():
    for mu, want in [(1.0, 0.1269), (0.0, 2.1269), (0.5, 1.1269)]:
        got = preference_loss(np.array([3.0]), np.array([1.0]), np.array([mu]))
        assert abs(got - want) < 1e-4, f"mu={mu}: expected {want}, got {got}"


def test_loss_is_a_mean_over_the_batch():
    r1, r2 = np.array([3.0, 3.0]), np.array([1.0, 1.0])
    got = preference_loss(r1, r2, np.array([1.0, 0.0]))
    assert abs(got - 1.1269) < 1e-4, f"expected mean(0.1269, 2.1269) = 1.1269, got {got}"


def test_tie_label_is_not_rounded():
    low = preference_loss(np.array([2.0]), np.array([0.0]), np.array([0.5]))
    hard = preference_loss(np.array([2.0]), np.array([0.0]), np.array([1.0]))
    assert low > hard + 0.5, f"a tie on an unequal pair must cost more than a correct hard label; got {low} and {hard}"
    even = preference_loss(np.array([4.0]), np.array([4.0]), np.array([0.5]))
    assert abs(even - np.log(2)) < 1e-9, f"equal returns with a tie must give log 2 = 0.6931, got {even}"


def test_large_gap_is_finite_and_exact():
    got = preference_loss(np.array([1000.0]), np.array([0.0]), np.array([0.0]))
    assert np.isfinite(got) and abs(got - 1000.0) < 1e-6, f"expected 1000.0 for a wrong label at gap 1000, got {got}"
    ok = preference_loss(np.array([0.0]), np.array([1000.0]), np.array([0.0]))
    assert np.isfinite(ok) and abs(ok) < 1e-6, f"expected 0 for a right label at gap 1000, got {ok}"


def test_eps_caps_the_loss():
    got = preference_loss(np.array([50.0]), np.array([0.0]), np.array([0.0]), eps=0.1)
    assert abs(got - 2.9957) < 1e-4, f"expected -log(0.05) = 2.9957 for a wrong label, got {got}"
