import numpy as np
from submission import unroll, unrolled_loss

PARAMS = {
    "H": np.eye(2),
    "G": np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 2.0]]),
    "Wp": np.eye(2),
    "wv": np.array([1.0, 1.0]),
}
OBS = np.array([1.0, 2.0])
ACTIONS = [0, 1]
PI = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])]
Z = [3.0, 5.0, 7.0]
U = [99.0, 1.0, 1.0]


def test_unroll_hand_computed_states():
    # s0 = [1, 2]. Action 0: r1 = 1, s1 = [2, 2]. Action 1: r2 = 2, s2 = [2, 4].
    r, p, v = unroll(PARAMS, OBS, ACTIONS)
    assert len(r) == 2, f"expected 2 rewards, got {len(r)}"
    assert len(p) == 3 and len(v) == 3, f"expected 3 policies and values, got {len(p)} and {len(v)}"
    assert np.allclose(r, [1.0, 2.0]), f"expected rewards [1, 2], got {r}"
    assert np.allclose(v, [3.0, 4.0, 6.0]), f"expected values [3, 4, 6], got {v}"


def test_unroll_policies_are_softmax_of_the_state():
    _, p, _ = unroll(PARAMS, OBS, ACTIONS)
    want = [[0.26894142, 0.73105858], [0.5, 0.5], [0.11920292, 0.88079708]]
    for k in range(3):
        assert np.allclose(p[k], want[k], atol=1e-6), f"step {k}: expected {want[k]}, got {p[k]}"


def test_the_action_order_matters():
    r, _, v = unroll(PARAMS, OBS, [1, 0])
    # Action 1 first: x = [1, 2, 0, 1], out = [1, 1, 4], s1 = [1, 4]; action 0: out = [1, 2, 4], s2 = [2, 4].
    assert np.allclose(r, [1.0, 1.0]), f"expected rewards [1, 1], got {r}"
    assert np.allclose(v, [3.0, 5.0, 6.0]), f"expected values [3, 5, 6], got {v}"


def test_zero_actions_is_a_single_prediction():
    r, p, v = unroll(PARAMS, OBS, [])
    assert len(r) == 0 and len(p) == 1 and len(v) == 1, f"expected 0, 1, 1 entries, got {len(r)}, {len(p)}, {len(v)}"


def test_loss_with_zero_parameters_is_hand_computable():
    zero = {"H": np.zeros((2, 2)), "G": np.zeros((3, 4)), "Wp": np.zeros((2, 2)), "wv": np.zeros(2)}
    # Every p is uniform, every v and r is 0. Value terms: 1 + 1 + 0.25. Policy: 3 log 2. Rewards (k >= 1): 4 + 9.
    want = 2.25 + 3 * np.log(2.0) + 13.0
    got = unrolled_loss(zero, OBS, ACTIONS, PI, [1.0, -1.0, 0.5], [9.0, 2.0, 3.0])
    assert abs(got - want) < 1e-9, f"expected {want}, got {got}"


def test_step_zero_has_no_reward_loss():
    zero = {"H": np.zeros((2, 2)), "G": np.zeros((3, 4)), "Wp": np.zeros((2, 2)), "wv": np.zeros(2)}
    a = unrolled_loss(zero, OBS, ACTIONS, PI, Z, [0.0, 1.0, 1.0])
    b = unrolled_loss(zero, OBS, ACTIONS, PI, Z, [1000.0, 1.0, 1.0])
    assert abs(a - b) < 1e-9, f"u[0] must not enter the loss, got {a} and {b}"


def test_loss_on_hand_computed_model():
    # k=0: 0 + 1.31326. k=1: 0 + 1 + 0.69315. k=2: 1 + 1 + 1.12693. Total 6.13334.
    got = unrolled_loss(PARAMS, OBS, ACTIONS, PI, Z, U)
    assert abs(got - 6.1333369) < 1e-5, f"expected 6.1333369, got {got}"


def test_policy_term_is_cross_entropy_with_the_full_target():
    # Same model, K = 0, target [0.5, 0.5], z = v so only the policy term remains.
    got = unrolled_loss(PARAMS, OBS, [], [np.array([0.5, 0.5])], [3.0], [0.0])
    want = -(0.5 * np.log(0.26894142) + 0.5 * np.log(0.73105858))
    assert abs(got - want) < 1e-6, f"expected {want}, got {got}"


def test_loss_is_zero_only_if_value_and_reward_match_and_policy_equals_entropy_floor():
    # With a one-hot target that matches a one-hot policy the loss reaches 0 only in the limit.
    # Here values and rewards match exactly, so the loss equals the policy cross-entropy alone.
    r, p, v = unroll(PARAMS, OBS, ACTIONS)
    pi = [np.array(x) for x in p]
    got = unrolled_loss(PARAMS, OBS, ACTIONS, pi, v, [0.0] + r)
    want = sum(-float(q @ np.log(q)) for q in p)
    assert abs(got - want) < 1e-9, f"expected the entropy sum {want}, got {got}"


def test_heads_read_the_state_with_their_own_weights():
    # Wp and wv differ, so mixing them up gives other numbers. s0 = H @ obs = [2, 1].
    params = {
        "H": np.eye(2),
        "G": np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
        "Wp": np.array([[1.0, 1.0], [0.0, 1.0]]),
        "wv": np.array([2.0, -1.0]),
    }
    r, p, v = unroll(params, np.array([2.0, 1.0]), [1])
    # step 0: logits [3, 1], v = 3. g: x = [2, 1, 0, 1], out = [1, 3, 1], r1 = 1, s1 = [3, 1].
    # step 1: logits [4, 1], v = 5.
    e0 = np.exp([3.0, 1.0] - np.max([3.0, 1.0]))
    e1 = np.exp([4.0, 1.0] - np.max([4.0, 1.0]))
    assert np.allclose(r, [1.0]), f"expected reward [1], got {r}"
    assert np.allclose(v, [3.0, 5.0]), f"expected values [3, 5], got {v}"
    assert np.allclose(p[0], e0 / e0.sum()), f"expected {e0 / e0.sum()}, got {p[0]}"
    assert np.allclose(p[1], e1 / e1.sum()), f"expected {e1 / e1.sum()}, got {p[1]}"
