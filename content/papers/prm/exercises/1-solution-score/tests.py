import numpy as np
from submission import solution_score, step_scores


def raises_value_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    except Exception:
        return False
    return False


def test_step_scores_neutral_as_positive():
    got = step_scores([0.5, 0.9], [0.3, 0.05])
    want = np.array([0.8, 0.95])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_step_scores_neutral_as_negative():
    got = step_scores([0.5, 0.9], [0.3, 0.05], neutral="negative")
    want = np.array([0.5, 0.9])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_product_by_hand():
    got = solution_score([0.95, 0.90, 0.92], [0.0, 0.0, 0.0])
    want = 0.95 * 0.90 * 0.92
    assert abs(got - want) < 1e-12, f"expected {want:.6f}, got {got}"
    assert isinstance(got, float), f"expected a Python float, got {type(got).__name__}"


def test_min_by_hand():
    got = solution_score([0.99, 0.40, 0.99], [0.0, 0.0, 0.0], reduction="min")
    assert abs(got - 0.40) < 1e-12, f"expected 0.4, got {got}"
    assert isinstance(got, float), f"expected a Python float, got {type(got).__name__}"


def test_neutral_changes_the_solution_score():
    pos, neu = [0.5] * 4, [0.3] * 4
    as_pos = solution_score(pos, neu)
    as_neg = solution_score(pos, neu, neutral="negative")
    assert abs(as_pos - 0.8**4) < 1e-12, f"expected {0.8**4:.4f}, got {as_pos}"
    assert abs(as_neg - 0.5**4) < 1e-12, f"expected {0.5**4:.4f}, got {as_neg}"


def test_product_penalizes_length_and_min_does_not():
    short = ([0.90, 0.95], [0.0, 0.0])
    long_ = ([0.98] * 10, [0.0] * 10)
    assert solution_score(*short) > solution_score(*long_), (
        f"product should prefer the short solution: short {solution_score(*short):.4f}, "
        f"long {solution_score(*long_):.4f}"
    )
    assert solution_score(*short, reduction="min") < solution_score(*long_, reduction="min"), (
        "min should prefer the long solution: "
        f"short {solution_score(*short, reduction='min'):.4f}, "
        f"long {solution_score(*long_, reduction='min'):.4f}"
    )


def test_single_step_is_the_same_for_both_reductions():
    a = solution_score([0.7], [0.1])
    b = solution_score([0.7], [0.1], reduction="min")
    assert abs(a - 0.8) < 1e-12 and abs(b - 0.8) < 1e-12, f"expected 0.8 twice, got {a} and {b}"


def test_inputs_are_not_modified():
    pos, neu = np.array([0.5, 0.6]), np.array([0.2, 0.1])
    pos0, neu0 = pos.copy(), neu.copy()
    step_scores(pos, neu)
    step_scores(pos, neu, neutral="negative")
    solution_score(pos, neu)
    assert np.array_equal(pos, pos0), f"p_pos changed: {pos0} -> {pos}"
    assert np.array_equal(neu, neu0), f"p_neu changed: {neu0} -> {neu}"


def test_invalid_arguments_raise_value_error():
    assert raises_value_error(step_scores, [0.5], [0.1], neutral="maybe"), "expected ValueError for neutral='maybe'"
    assert raises_value_error(solution_score, [0.5], [0.1], reduction="mean"), "expected ValueError for reduction='mean'"
    assert raises_value_error(solution_score, [], []), "expected ValueError for an empty solution"
