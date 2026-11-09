import numpy as np

from submission import ctc_backward, ctc_forward, ctc_grad

Y = np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])  # columns: blank, a, b


def random_y(seed, T, K):
    rng = np.random.default_rng(seed)
    y = rng.random((T, K)) + 0.05
    return y / y.sum(axis=1, keepdims=True)


def softmax(u):
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_beta_table_for_ab():
    beta = ctc_backward(Y, (1, 2))
    want = np.array([
        [0.048, 0.225, 0.132, 0.039, 0.06],
        [0.0, 0.12, 0.09, 0.24, 0.15],
        [0.0, 0.0, 0.0, 0.3, 0.5],
    ])
    assert beta.shape == (3, 5), f"beta shape: expected (3, 5), got {beta.shape}"
    assert np.allclose(beta, want), f"beta: expected\n{want}\ngot\n{beta}"


def test_equation_14_holds_at_every_frame():
    for seed, (T, K, label) in enumerate([(5, 3, (1, 2)), (6, 3, (2, 1, 2)), (6, 4, (3, 1, 1)), (5, 3, (1, 1))]):
        y = random_y(seed, T, K)
        alpha, p = ctc_forward(y, label)
        beta = ctc_backward(y, label)
        lp = [0]
        for k in label:
            lp += [k, 0]
        for t in range(T):
            total = sum(alpha[t, s] * beta[t, s] / y[t, lp[s]] for s in range(len(lp)))
            assert abs(total - p) < 1e-12, f"label={label}, t={t}: sum_s alpha*beta/y should equal p={p}, got {total}"


def test_beta_respects_the_repeat_rule():
    # For 'aa', the backward skip from the first a to the second a is forbidden.
    y = random_y(3, 4, 3)
    alpha, p = ctc_forward(y, (1, 1))
    beta = ctc_backward(y, (1, 1))
    total = sum(alpha[0, s] * beta[0, s] / y[0, [0, 1, 0, 1, 0][s]] for s in range(5))
    assert abs(total - p) < 1e-12, f"'aa': expected p={p}, got {total}; check the beta skip condition"


def test_gradient_matches_finite_differences():
    for seed, (T, K, label) in enumerate([(4, 3, (1, 2)), (5, 3, (1, 1)), (5, 4, (2, 3, 1))]):
        rng = np.random.default_rng(10 + seed)
        u = rng.normal(size=(T, K))

        def loss(u):
            return -np.log(ctc_forward(softmax(u), label)[1])

        want = np.zeros_like(u)
        for i in np.ndindex(*u.shape):
            d = np.zeros_like(u)
            d[i] = 1e-6
            want[i] = (loss(u + d) - loss(u - d)) / 2e-6
        got = ctc_grad(softmax(u), label)
        assert got.shape == (T, K), f"gradient shape: expected {(T, K)}, got {got.shape}"
        assert np.allclose(got, want, atol=1e-6), f"label={label}: expected\n{want}\ngot\n{got}"


def test_gradient_page_example():
    got = ctc_grad(Y, (1, 2))
    want = np.array([[0.2242, -0.3242, 0.1], [0.1352, 0.0044, -0.1396], [0.2253, 0.2, -0.4253]])
    assert np.allclose(got, want, atol=1e-4), f"expected\n{want}\ngot\n{got}"


def test_each_frame_gradient_sums_to_zero():
    # y sums to 1 and the posterior over symbols at frame t sums to 1, so the rows cancel.
    y = random_y(5, 6, 4)
    got = ctc_grad(y, (1, 2, 3))
    assert np.allclose(got.sum(axis=1), 0.0, atol=1e-10), f"row sums should be 0, got {got.sum(axis=1)}"
