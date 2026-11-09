import numpy as np
from submission import train_lora, trainable_params, lora_forward

d, k, n = 12, 12, 200
rng = np.random.default_rng(0)
W0 = rng.normal(size=(d, k)) / np.sqrt(k)
U = rng.normal(size=(d, 2))
V = rng.normal(size=(2, k))
DELTA = 0.5 * (U @ V) / np.sqrt(k)
X = rng.normal(size=(n, k))
Y = X @ (W0 + DELTA).T
BASE_LOSS = float(np.sum((X @ W0.T - Y) ** 2) / (2 * n))


def test_trainable_params():
    assert trainable_params(12, 12, 2) == 48, f"expected 2 * (12 + 12) = 48, got {trainable_params(12, 12, 2)}"
    assert trainable_params(4, 7, 3) == 33, f"expected 3 * (4 + 7) = 33, got {trainable_params(4, 7, 3)}"
    full = 12 * 12
    assert trainable_params(12, 12, 2) < full, "rank 2 must be smaller than a full 12 x 12 update"


def test_loss_curve_starts_at_the_base_model():
    A, B, losses = train_lora(X, Y, W0, r=2, alpha=2.0, steps=5, lr=0.1)
    assert len(losses) == 6, f"expected steps + 1 = 6 losses, got {len(losses)}"
    assert abs(losses[0] - BASE_LOSS) < 1e-9, f"with B = 0 the first loss must equal the base loss {BASE_LOSS:.6f}, got {losses[0]:.6f}"
    assert A.shape == (2, k) and B.shape == (d, 2), f"expected A (2, {k}) and B ({d}, 2), got {A.shape} and {B.shape}"


def test_rank_two_fits_a_rank_two_target():
    _, _, losses = train_lora(X, Y, W0, r=2, alpha=2.0, steps=1500, lr=0.2)
    assert losses[-1] < 1e-3 * BASE_LOSS, f"rank 2 should reach under 0.1% of the base loss {BASE_LOSS:.4f}, got {losses[-1]:.6f}"
    assert losses[-1] < losses[0], "the loss must go down"


def test_rank_one_cannot_fit_a_rank_two_target():
    _, _, losses = train_lora(X, Y, W0, r=1, alpha=1.0, steps=1500, lr=0.2)
    s = np.linalg.svd(DELTA, compute_uv=False)
    floor = 0.5 * s[1] ** 2 * np.mean(np.sum(X**2, axis=1)) / k
    assert losses[-1] > 0.5 * floor, f"rank 1 cannot go below about {floor:.4f}; got {losses[-1]:.6f}"
    assert losses[-1] < 0.9 * BASE_LOSS, f"rank 1 should still reduce the loss {BASE_LOSS:.4f}, got {losses[-1]:.4f}"


def test_the_learned_update_has_rank_at_most_r():
    A, B, _ = train_lora(X, Y, W0, r=2, alpha=2.0, steps=50, lr=0.2)
    rank = np.linalg.matrix_rank(B @ A, tol=1e-9)
    assert rank <= 2, f"B A has rank {rank}, expected at most 2"


def test_w0_is_frozen():
    W = W0.copy()
    train_lora(X, Y, W, r=2, alpha=2.0, steps=20, lr=0.2)
    assert np.array_equal(W, W0), "train_lora must not modify W0"


def test_trained_layer_matches_target_outputs():
    A, B, _ = train_lora(X, Y, W0, r=2, alpha=2.0, steps=1500, lr=0.2)
    H = lora_forward(X, W0, A, B, 2.0)
    err = np.abs(H - Y).max()
    assert err < 0.05, f"outputs of the trained layer should match the target; max abs error {err:.4f}"
