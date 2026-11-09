import numpy as np
from submission import expert_fractions, load_balancing_loss

# Router probabilities from the page: 6 tokens, 3 experts. Four tokens pick
# expert 0, one picks expert 1, one picks expert 2.
P6 = np.array([
    [0.7870, 0.1065, 0.1065],
    [0.6285, 0.2312, 0.1402],
    [0.5065, 0.1863, 0.3072],
    [0.6652, 0.2447, 0.0900],
    [0.1065, 0.7870, 0.1065],
    [0.1065, 0.1065, 0.7870],
])


def test_fractions_on_the_page_example():
    f, P = expert_fractions(P6)
    assert np.allclose(f, [4 / 6, 1 / 6, 1 / 6]), f"f expected [0.6667 0.1667 0.1667], got {f}"
    assert np.allclose(P, [0.4667, 0.2770, 0.2562], atol=1e-3), f"P expected [0.4667 0.277 0.2562], got {P}"


def test_f_uses_hard_assignments():
    # Token 0 splits its mass 0.4 / 0.6, so its hard choice is expert 1.
    p = np.array([[0.4, 0.6], [0.4, 0.6], [0.9, 0.1]])
    f, P = expert_fractions(p)
    assert np.allclose(f, [1 / 3, 2 / 3]), f"f must count argmax choices: expected [0.3333 0.6667], got {f}"
    assert np.allclose(P, [0.5667, 0.4333], atol=1e-3), f"P must average probabilities: expected [0.5667 0.4333], got {P}"


def test_uniform_routing_gives_alpha():
    for N in (2, 4, 8):
        p = np.full((N * 4, N), 1 / N)
        # Break the argmax ties evenly: tokens cycle through the experts.
        p = p + 1e-9 * np.eye(N)[np.arange(N * 4) % N]
        loss = load_balancing_loss(p, alpha=0.01)
        assert np.isclose(loss, 0.01, atol=1e-6), f"N={N}: balanced routing should give alpha = 0.01, got {loss}"


def test_collapse_gives_alpha_times_n():
    N = 4
    p = np.zeros((8, N))
    p[:, 0] = 1.0
    loss = load_balancing_loss(p, alpha=0.01)
    assert np.isclose(loss, 0.04), f"all tokens on one expert should give alpha * N = 0.04, got {loss}"


def test_page_example_loss():
    loss = load_balancing_loss(P6, alpha=1e-2)
    assert np.isclose(loss, 0.012001, atol=2e-5), f"expected 0.012001, got {loss}"


def test_default_alpha_and_scalar():
    loss = load_balancing_loss(P6)
    assert np.isscalar(loss) or np.ndim(loss) == 0, f"expected a scalar, got shape {np.shape(loss)}"
    assert np.isclose(loss, 0.012001, atol=2e-5), f"default alpha should be 1e-2: expected 0.012001, got {loss}"


def test_batch_size_does_not_change_the_loss():
    big = np.tile(P6, (10, 1))
    a, b = load_balancing_loss(P6), load_balancing_loss(big)
    assert np.isclose(a, b), f"f and P are means, so repeating the batch must not change the loss: {a} vs {b}"


def test_balanced_is_lower_than_skewed():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 4))
    skew = logits + np.array([3.0, 0, 0, 0])

    def sm(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    bal, sk = load_balancing_loss(sm(logits)), load_balancing_loss(sm(skew))
    assert sk > bal, f"a router that favors expert 0 should cost more: balanced {bal}, skewed {sk}"


def test_alpha_scales_the_loss():
    a, b = load_balancing_loss(P6, alpha=0.5), load_balancing_loss(P6, alpha=0.25)
    assert np.isclose(a, 0.6000, atol=1e-3), f"alpha=0.5: expected 0.6 (= 0.5 * 3 * 0.4), got {a}"
    assert np.isclose(a, 2 * b), f"the loss must be linear in alpha: expected {2 * b}, got {a}"
