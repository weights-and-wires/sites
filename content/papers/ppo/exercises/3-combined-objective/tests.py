import numpy as np
from submission import entropy, ppo_loss, ppo_objective

probs_old = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.1, 0.1, 0.8]])
probs = np.array([[0.65, 0.2, 0.15], [0.3, 0.5, 0.2], [0.1, 0.1, 0.8]])
actions = np.array([0, 1, 2])
adv = np.array([1.0, -0.5, 2.0])
values = np.array([1.0, 0.0, 2.5])
returns = np.array([1.5, 0.5, 2.0])


def test_entropy_values():
    got = entropy(np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0 + 1e-300]]))
    assert abs(got[0] - np.log(3)) < 1e-9, f"uniform over 3 expected ln 3 = {np.log(3)}, got {got[0]}"
    assert got.shape == (2,), f"expected one entropy per row, shape (2,), got {got.shape}"
    assert abs(got[1] - np.log(2)) < 1e-6, f"[0.5, 0.5, 0] expected ln 2 = {np.log(2)}, got {got[1]}"


def test_entropy_of_rows_from_page():
    got = entropy(probs)
    want = [0.8865, 1.0297, 0.639]
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_objective_from_page():
    got = ppo_objective(probs, probs_old, actions, adv, values, returns, 0.2, 0.5, 0.01)
    assert abs(got - 0.7419) < 1e-4, f"expected 0.7419, got {got}"


def test_loss_is_negative_objective():
    args = (probs, probs_old, actions, adv, values, returns, 0.2, 0.5, 0.01)
    assert abs(ppo_loss(*args) + ppo_objective(*args)) < 1e-12, f"loss must equal -objective; got {ppo_loss(*args)} and {ppo_objective(*args)}"


def test_each_coefficient_has_the_right_sign():
    base = ppo_objective(probs, probs_old, actions, adv, values, returns, 0.2, 0.0, 0.0)
    assert abs(base - 0.8583) < 1e-4, f"with c1=c2=0 expected the mean L_CLIP 0.8583, got {base}"
    vf = ppo_objective(probs, probs_old, actions, adv, values, returns, 0.2, 1.0, 0.0)
    assert abs(vf - (base - 0.25)) < 1e-9, f"c1=1 subtracts the mean squared error 0.25: expected {base - 0.25}, got {vf}"
    ent = ppo_objective(probs, probs_old, actions, adv, values, returns, 0.2, 0.0, 1.0)
    assert abs(ent - (base + 0.8517)) < 1e-4, f"c2=1 adds the mean entropy 0.8517: expected {base + 0.8517}, got {ent}"


def test_value_error_is_squared():
    v2 = values + 2.0
    got = ppo_objective(probs, probs_old, actions, adv, v2, returns, 0.2, 1.0, 0.0)
    base = ppo_objective(probs, probs_old, actions, adv, values, returns, 0.2, 0.0, 0.0)
    want = base - np.mean((v2 - returns) ** 2)
    assert abs(got - want) < 1e-9, f"expected {want} (squared error), got {got}"


def test_ratio_uses_chosen_action_and_clips():
    # Action 0 in row 0: r = 1.3 > 1.2 and A > 0, so the term is 1.2 and not 1.3.
    p = np.array([[0.65, 0.35]])
    po = np.array([[0.5, 0.5]])
    got = ppo_objective(p, po, np.array([0]), np.array([1.0]), np.array([0.0]), np.array([0.0]), 0.2, 0.0, 0.0)
    assert abs(got - 1.2) < 1e-9, f"expected 1.2 (clipped), got {got}"
    got = ppo_objective(p, po, np.array([1]), np.array([1.0]), np.array([0.0]), np.array([0.0]), 0.2, 0.0, 0.0)
    assert abs(got - 0.7) < 1e-9, f"action 1 has r = 0.35/0.5 = 0.7 and A>0, which min keeps at 0.7; got {got}"
