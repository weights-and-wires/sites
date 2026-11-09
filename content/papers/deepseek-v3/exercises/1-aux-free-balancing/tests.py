import numpy as np

from submission import select_experts, gate_values, expert_load, update_bias

S = np.array([[0.9, 0.8, 0.3, 0.2]])
B = np.array([-0.5, 0.0, 0.2, 0.0])


def test_zero_bias_is_plain_topk():
    idx = np.sort(select_experts(S, np.zeros(4), 2), axis=1)
    assert idx.tolist() == [[0, 1]], f"expected experts [[0, 1]], got {idx.tolist()}"


def test_bias_changes_selection():
    idx = np.sort(select_experts(S, B, 2), axis=1)
    assert idx.tolist() == [[1, 2]], f"s + b = [0.4, 0.8, 0.5, 0.2], expected experts [[1, 2]], got {idx.tolist()}"


def test_select_shape_and_rows():
    rng = np.random.default_rng(1)
    s = rng.random((5, 10))
    b = rng.normal(0, 0.1, 10)
    idx = select_experts(s, b, 3)
    assert idx.shape == (5, 3), f"expected shape (5, 3), got {idx.shape}"
    for t in range(5):
        want = set(np.argsort(-(s[t] + b))[:3].tolist())
        assert set(idx[t].tolist()) == want, f"row {t}: expected {sorted(want)}, got {sorted(idx[t].tolist())}"


def test_gates_use_original_affinity():
    idx = np.array([[1, 2]])
    g = gate_values(S, idx)
    want = np.array([[0.8, 0.3]]) / 1.1
    assert np.allclose(g, want), f"expected {want} (from s, not s + b), got {g}"


def test_gates_sum_to_one_and_follow_idx_order():
    rng = np.random.default_rng(2)
    s = rng.random((4, 8))
    idx = np.argsort(-s, axis=1)[:, :3]
    g = gate_values(s, idx)
    assert np.allclose(g.sum(axis=1), 1.0), f"gates must sum to 1 per token, got {g.sum(axis=1)}"
    want = np.take_along_axis(s, idx, axis=1)
    want = want / want.sum(axis=1, keepdims=True)
    assert np.allclose(g, want), f"expected {want}, got {g}"


def test_expert_load():
    idx = np.array([[0, 1], [1, 2], [1, 3]])
    got = expert_load(idx, 5)
    assert got.tolist() == [1, 3, 1, 1, 0], f"expected [1, 3, 1, 1, 0], got {got.tolist()}"


def test_update_bias_direction():
    b = np.zeros(4)
    load = np.array([10, 6, 4, 4])  # mean 6
    got = update_bias(b, load, 0.01)
    want = np.array([-0.01, 0.0, 0.01, 0.01])
    assert np.allclose(got, want), f"overloaded down, underloaded up, at-mean unchanged: expected {want}, got {got}"


def test_update_bias_step_size_ignores_how_far_off():
    got = update_bias(np.zeros(3), np.array([100, 7, 1]), 0.5)  # mean 36
    want = np.array([-0.5, 0.5, 0.5])
    assert np.allclose(got, want), f"the step is gamma, not proportional to the load gap: expected {want}, got {got}"


def test_balancing_loop_reduces_imbalance():
    rng = np.random.default_rng(0)
    n, k, tokens = 16, 2, 512
    pref = rng.normal(0, 0.8, n)
    b = np.zeros(n)
    first, last = None, None
    for step in range(300):
        s = 1 / (1 + np.exp(-(rng.normal(0, 1, (tokens, n)) + pref)))
        load = expert_load(select_experts(s, b, k), n)
        cv = load.std() / load.mean()
        if step == 0:
            first = cv
        last = cv
        b = update_bias(b, load, 0.005)
    assert first > 0.5, f"the toy router should start imbalanced, got CV {first:.3f}"
    assert last < 0.2, f"expected load CV below 0.2 after 300 steps, got {last:.3f} (started at {first:.3f})"
