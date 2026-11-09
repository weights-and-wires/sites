import numpy as np
from submission import build_tokens, state_token_indices, causal_self_attention

K, D = 3, 4


def make(seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(K, D)) for _ in range(4)]


def test_shape_and_order_without_timestep():
    R = np.full((K, D), 1.0)
    s = np.full((K, D), 2.0)
    a = np.full((K, D), 3.0)
    for t in range(K):
        R[t] += 10 * t
        s[t] += 10 * t
        a[t] += 10 * t
    X = build_tokens(R, s, a, np.zeros((K, D)))
    assert X.shape == (3 * K, D), f"expected shape {(3 * K, D)}, got {X.shape}"
    want = [1, 2, 3, 11, 12, 13, 21, 22, 23]
    assert np.allclose(X[:, 0], want), f"expected the order R1 s1 a1 R2 s2 a2 ... = {want}, got {X[:, 0].tolist()}"


def test_timestep_embedding_added_to_all_three_tokens():
    R, s, a, t = make()
    X = build_tokens(R, s, a, t)
    for i in range(K):
        for j, base in enumerate((R, s, a)):
            want = base[i] + t[i]
            got = X[3 * i + j]
            assert np.allclose(got, want), f"token {3 * i + j} (timestep {i}, modality {'Rsa'[j]}) expected {want}, got {got}"


def test_timestep_embedding_is_per_timestep_not_per_token():
    R, s, a, _ = make()
    t = np.zeros((K, D))
    t[1] = 7.0
    X = build_tokens(R, s, a, t)
    base = build_tokens(R, s, a, np.zeros((K, D)))
    diff = (X - base)[:, 0]
    assert np.allclose(diff, [0, 0, 0, 7, 7, 7, 0, 0, 0]), f"only the three tokens of timestep 1 should shift by 7, got shifts {diff.tolist()}"


def test_state_token_indices():
    got = state_token_indices(4)
    assert np.array_equal(got, [1, 4, 7, 10]), f"expected [1, 4, 7, 10], got {got}"
    assert np.issubdtype(np.asarray(got).dtype, np.integer), f"expected integer positions, got dtype {np.asarray(got).dtype}"


def test_state_indices_select_state_tokens():
    R, s, a, t = make(3)
    X = build_tokens(R, s, a, t)
    got = X[state_token_indices(K)]
    assert np.allclose(got, s + t), f"expected the state tokens s + t, got {got}"


def test_hidden_state_at_s_t_cannot_see_a_t():
    R, s, a, t = make(5)
    base = causal_self_attention(build_tokens(R, s, a, t))
    idx = state_token_indices(K)
    for step in range(K):
        a2 = a.copy()
        a2[step:] += 5.0  # change a_step and every later action
        out = causal_self_attention(build_tokens(R, s, a2, t))
        assert np.allclose(out[idx[step]], base[idx[step]]), (
            f"the prediction for a_{step} must not depend on a_{step} or later actions; "
            f"hidden state at s_{step} changed by {np.abs(out[idx[step]] - base[idx[step]]).max():.4f}"
        )


def test_hidden_state_at_s_t_sees_return_state_and_earlier_action():
    R, s, a, t = make(7)
    base = causal_self_attention(build_tokens(R, s, a, t))
    idx = state_token_indices(K)
    step = 2
    bump = np.zeros((K, D))
    bump[step] = 3.0
    bump_prev = np.zeros((K, D))
    bump_prev[1] = 3.0
    for name, args in (("R_2", (R + bump, s, a)), ("s_2", (R, s + bump, a)), ("a_1", (R, s, a + bump_prev))):
        out = causal_self_attention(build_tokens(*args, t))
        assert not np.allclose(out[idx[step]], base[idx[step]]), f"the hidden state at s_{step} must depend on {name}, but it did not change"
