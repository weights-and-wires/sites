import numpy as np
from submission import soft_update, updates_to_shrink


def make():
    target = {"W": np.zeros((2, 2)), "b": np.zeros(2)}
    source = {"W": np.ones((2, 2)), "b": np.full(2, 2.0)}
    return target, source


def test_one_step_with_tau_point_one():
    target, source = make()
    new = soft_update(target, source, 0.1)
    assert np.allclose(new["W"], 0.1), f"expected W = 0.1 everywhere, got {new['W']}"
    assert np.allclose(new["b"], 0.2), f"expected b = 0.2, got {new['b']}"


def test_three_steps_match_the_page():
    target, source = make()
    for _ in range(3):
        target = soft_update(target, source, 0.1)
    assert np.allclose(target["W"], 0.271), f"expected 0.271 after 3 updates, got {target['W']}"


def test_weights_are_not_swapped():
    target = {"w": np.array([10.0])}
    source = {"w": np.array([0.0])}
    new = soft_update(target, source, 0.25)
    assert np.allclose(new["w"], 7.5), f"expected 0.25 * 0 + 0.75 * 10 = 7.5, got {new['w']}"


def test_tau_one_copies_and_tau_zero_keeps():
    target, source = make()
    new = soft_update(target, source, 1.0)
    assert np.allclose(new["b"], source["b"]), f"tau = 1 must copy the source, got {new['b']}"
    new = soft_update(target, source, 0.0)
    assert np.allclose(new["b"], target["b"]), f"tau = 0 must keep the target, got {new['b']}"


def test_inputs_are_not_modified():
    target, source = make()
    soft_update(target, source, 0.5)
    assert np.allclose(target["W"], 0.0), f"the target dict changed: {target['W']}"
    assert np.allclose(source["W"], 1.0), f"the source dict changed: {source['W']}"


def test_result_does_not_alias_the_inputs():
    target, source = make()
    new = soft_update(target, source, 1.0)
    new["W"][0, 0] = 99.0
    assert source["W"][0, 0] == 1.0, "the returned arrays must not share memory with the source"


def test_updates_to_shrink_on_page_values():
    got = [updates_to_shrink(0.001, f) for f in (0.5, 0.1, 0.01)]
    assert got == [693, 2302, 4603], f"expected [693, 2302, 4603], got {got}"


def test_updates_to_shrink_is_the_smallest_n():
    for tau, frac in ((0.1, 0.5), (0.5, 0.1), (0.01, 0.3)):
        n = updates_to_shrink(tau, frac)
        assert (1 - tau) ** n <= frac, f"n = {n} does not reach the fraction {frac}"
        assert (1 - tau) ** (n - 1) > frac, f"n = {n} is not the smallest: n - 1 also reaches {frac}"
