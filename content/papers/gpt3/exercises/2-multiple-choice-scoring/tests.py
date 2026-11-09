import numpy as np
from submission import choose, score

OPTIONS = [
    {"ctx": [-1.5], "unc": [-6.0]},
    {"ctx": [-0.5, -0.6, -0.5, -0.4], "unc": [-0.4, -0.5, -0.5, -0.4]},
    {"ctx": [-2.5], "unc": [-6.5]},
]


def test_sum():
    got = [score(o, "sum") for o in OPTIONS]
    assert np.allclose(got, [-1.5, -2.0, -2.5]), f"expected [-1.5, -2.0, -2.5], got {got}"


def test_mean():
    got = [score(o, "mean") for o in OPTIONS]
    assert np.allclose(got, [-1.5, -0.5, -2.5]), f"expected [-1.5, -0.5, -2.5], got {got}"


def test_ratio_is_a_difference_of_totals():
    got = [score(o, "ratio") for o in OPTIONS]
    assert np.allclose(got, [4.5, -0.2, 4.0]), f"expected [4.5, -0.2, 4.0], got {got}"


def test_the_three_rules_can_disagree():
    picks = [choose(OPTIONS, m) for m in ("sum", "mean", "ratio")]
    assert picks == [0, 1, 0], f"expected picks [0, 1, 0] for sum, mean, ratio; got {picks}"


def test_length_normalization_flips_the_choice():
    short = {"ctx": [-2.0], "unc": None}
    long = {"ctx": [-1.0, -1.0, -1.0], "unc": None}
    assert choose([short, long], "sum") == 0, "the total favors the short option (-2 beats -3)"
    assert choose([short, long], "mean") == 1, "the per-token average favors the long option (-1 beats -2)"


def test_ratio_prefers_the_option_the_context_raises():
    generic = {"ctx": [-1.0], "unc": [-1.0]}
    specific = {"ctx": [-3.0], "unc": [-9.0]}
    assert choose([generic, specific], "mean") == 0, "per-token likelihood prefers the generic option"
    assert choose([generic, specific], "ratio") == 1, "the ratio rewards the option the context made likelier (+6 vs 0)"


def test_ratio_does_not_divide_by_length():
    opt = {"ctx": [-1.0, -1.0, -1.0, -1.0], "unc": [-3.0, -3.0, -3.0, -3.0]}
    got = score(opt, "ratio")
    assert abs(got - 8.0) < 1e-9, f"expected the total gain of 8.0, got {got}"


def test_ties_take_the_first_index():
    opts = [{"ctx": [-1.0], "unc": None}, {"ctx": [-1.0], "unc": None}]
    assert choose(opts, "sum") == 0, f"expected index 0 on a tie, got {choose(opts, 'sum')}"


def test_errors():
    for bad in (lambda: score(OPTIONS[0], "max"), lambda: score({"ctx": [-1.0], "unc": None}, "ratio")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError for an unknown mode and for ratio without unc")
