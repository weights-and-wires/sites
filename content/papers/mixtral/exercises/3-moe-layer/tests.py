import numpy as np
from submission import moe_forward, top_k_gate

T, D, N = 12, 4, 8


def make(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, D))
    Wg = rng.normal(size=(D, N))
    mats = [rng.normal(size=(D, D)) for _ in range(N)]
    return x, Wg, mats


class Recorder:
    """A linear expert that records how many rows it receives."""

    def __init__(self, M):
        self.M, self.rows, self.calls = M, 0, 0

    def __call__(self, h):
        self.rows += h.shape[0]
        self.calls += 1
        return h @ self.M


def reference(x, Wg, mats, k=2):
    idx, w = top_k_gate(x @ Wg, k)
    y = np.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(k):
            y[t] += w[t, j] * (x[t] @ mats[idx[t, j]])
    return y


def test_matches_token_by_token_reference():
    x, Wg, mats = make()
    got = moe_forward(x, Wg, [Recorder(M) for M in mats])
    want = reference(x, Wg, mats)
    assert got.shape == x.shape, f"expected shape {x.shape}, got {got.shape}"
    assert np.allclose(got, want), f"expected row 0 {want[0]}, got {got[0]}"


def test_each_token_visits_exactly_k_experts():
    x, Wg, mats = make(1)
    experts = [Recorder(M) for M in mats]
    moe_forward(x, Wg, experts)
    seen = sum(e.rows for e in experts)
    assert seen == T * 2, f"the experts must see T*k = {T * 2} token rows in total, got {seen}"


def test_expert_sees_only_its_own_tokens():
    x, Wg, mats = make(2)
    experts = [Recorder(M) for M in mats]
    moe_forward(x, Wg, experts)
    idx, _ = top_k_gate(x @ Wg, 2)
    for e in range(N):
        want = int((idx == e).sum())
        assert experts[e].rows == want, f"expert {e} should receive {want} rows, got {experts[e].rows}"


def test_identity_experts_return_the_input():
    # Weights sum to 1 per token, so identity experts give y = x.
    # Unnormalized weights would shrink the output.
    x, Wg, _ = make(3)
    got = moe_forward(x, Wg, [lambda h: h for _ in range(N)])
    assert np.allclose(got, x), f"identity experts must give y = x, got max error {np.abs(got - x).max()}"


def test_constant_experts_show_the_weights():
    # Expert i returns i+1 in every coordinate, so y = w0*(i0+1) + w1*(i1+1).
    x, Wg, _ = make(4)
    experts = [(lambda h, c=i + 1.0: np.full_like(h, c)) for i in range(N)]
    got = moe_forward(x, Wg, experts)
    idx, w = top_k_gate(x @ Wg, 2)
    want = (w * (idx + 1.0)).sum(axis=1)
    assert np.allclose(got[:, 0], want), f"expected {want[:3]}, got {got[:3, 0]}"


def test_unused_expert_gets_no_rows():
    x = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    Wg = np.zeros((D, N))
    Wg[0, 0], Wg[1, 1] = 5.0, 5.0
    Wg[0, 2], Wg[1, 2] = 2.0, 2.0
    experts = [Recorder(np.eye(D)) for _ in range(N)]
    moe_forward(x, Wg, experts)
    assert sum(experts[e].rows for e in range(3, N)) == 0, f"experts 3 to 7 must receive no rows, got {[e.rows for e in experts]}"
