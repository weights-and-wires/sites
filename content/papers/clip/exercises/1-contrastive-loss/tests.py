import numpy as np
from submission import l2_normalize, clip_loss


def reference(I_f, T_f, scale):
    """Loop-based reference: for every pair, -log of the correct entry's share."""
    I_e = I_f / np.linalg.norm(I_f, axis=1, keepdims=True)
    T_e = T_f / np.linalg.norm(T_f, axis=1, keepdims=True)
    n = len(I_e)
    logits = scale * (I_e @ T_e.T)
    li = lt = 0.0
    for k in range(n):
        row = logits[k]
        col = logits[:, k]
        li += np.log(np.sum(np.exp(row - row.max()))) + row.max() - logits[k, k]
        lt += np.log(np.sum(np.exp(col - col.max()))) + col.max() - logits[k, k]
    return (li / n + lt / n) / 2


def test_l2_normalize_rows_and_axis():
    x = np.array([[3.0, 4.0], [0.0, 2.0]])
    out = l2_normalize(x, axis=1)
    assert np.allclose(out, [[0.6, 0.8], [0.0, 1.0]]), f"expected [[0.6, 0.8], [0, 1]], got {out}"
    cols = l2_normalize(x, axis=0)
    assert np.allclose(np.linalg.norm(cols, axis=0), 1), f"columns must have unit length for axis=0, got {np.linalg.norm(cols, axis=0)}"


def test_hand_example_from_the_page():
    # Cosine similarities [[1, 0.6], [0, 0.8]], scale 10, so logits [[10, 6], [0, 8]].
    I_f = np.eye(2)
    T_f = np.array([[1.0, 0.0], [0.6, 0.8]])
    loss = clip_loss(I_f, T_f, np.log(10.0))
    assert np.isclose(loss, 0.03636, atol=1e-4), f"expected 0.03636, got {loss}"


def test_features_are_normalized_inside():
    I_f = np.diag([2.0, 5.0])
    T_f = np.array([[3.0, 0.0], [1.5, 2.0]])  # row norms 3 and 2.5, directions as in the hand example
    loss = clip_loss(I_f, T_f, np.log(10.0))
    assert np.isclose(loss, 0.03636, atol=1e-4), f"expected 0.03636 after normalizing, got {loss}; normalize I_f and T_f first"


def test_identity_similarity():
    for n, want in [(4, 0.74367), (8, 1.27401)]:
        I = np.eye(n)
        loss = clip_loss(I, I, 0.0)
        assert np.isclose(loss, want, atol=1e-4), f"n={n}, t=0: expected ln(1+(n-1)/e) = {want}, got {loss}"


def test_loss_is_a_mean_over_the_batch():
    small = clip_loss(np.eye(4), np.eye(4), 0.0)
    big = clip_loss(np.eye(8), np.eye(8), 0.0)
    assert small < 1.0 and big < 1.5, f"a per-pair mean stays near ln n; got {small} for n=4 and {big} for n=8"


def test_both_directions_count():
    I_f = np.eye(2)
    T_f = np.array([[1.0, 0.0], [0.6, 0.8]])
    img_only, txt_only = 0.00924, 0.06349
    loss = clip_loss(I_f, T_f, np.log(10.0))
    assert abs(loss - img_only) > 1e-3, f"loss {loss} equals the image-to-text term alone ({img_only}); add the text-to-image term"
    assert abs(loss - txt_only) > 1e-3, f"loss {loss} equals the text-to-image term alone ({txt_only}); add the image-to-text term"
    assert np.isclose(loss, (img_only + txt_only) / 2, atol=1e-4), f"expected the average 0.03636, got {loss}"


def test_symmetric_in_the_two_modalities():
    rng = np.random.default_rng(0)
    I_f, T_f = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    a, b = clip_loss(I_f, T_f, 1.0), clip_loss(T_f, I_f, 1.0)
    assert np.isclose(a, b), f"swapping images and texts must not change the loss, got {a} and {b}"


def test_matches_reference_on_random_batches():
    rng = np.random.default_rng(1)
    for n, d, t in [(3, 4, 0.5), (7, 5, np.log(1 / 0.07)), (5, 8, np.log(100.0))]:
        I_f, T_f = rng.normal(size=(n, d)), rng.normal(size=(n, d))
        want = reference(I_f, T_f, np.exp(t))
        got = clip_loss(I_f, T_f, t)
        assert np.isclose(got, want, atol=1e-8), f"n={n}, t={t:.3f}: expected {want}, got {got}"


def test_scale_is_exp_of_t():
    rng = np.random.default_rng(2)
    I_f, T_f = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    got = clip_loss(I_f, T_f, 0.0)
    want = reference(I_f, T_f, 1.0)
    assert np.isclose(got, want), f"t=0 means scale exp(0)=1; expected {want}, got {got}"
    assert np.isclose(clip_loss(np.eye(3), np.eye(3), 0.0), np.log(1 + 2 / np.e)), "t=0 must give a logit scale of 1, not 0"


def test_stable_for_large_scale():
    # Pairs are mismatched, so the wrong logits are 1000 and the right ones are 0.
    I_f = np.eye(3)
    T_f = np.eye(3)[[1, 2, 0]]
    loss = clip_loss(I_f, T_f, np.log(1000.0))
    assert np.isfinite(loss), f"loss is {loss} at scale 1000; subtract the maximum before exp"
    assert np.isclose(loss, 1000.0, rtol=1e-6), f"expected 1000, got {loss}"
