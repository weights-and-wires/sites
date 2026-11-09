import numpy as np
from submission import train_dpo

REF = np.array([0.70, 0.25, 0.05])
R_STAR = np.array([0.0, 1.0, 2.0])


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def bt_counts(r, n=1000.0):
    C = n / (1.0 + np.exp(-(r[:, None] - r[None, :])))
    np.fill_diagonal(C, 0.0)
    return C


C = bt_counts(R_STAR)


def run(beta, steps=3000):
    ref_logits = np.log(REF)
    return train_dpo(ref_logits, C, beta, 8.0 / beta**2, steps)


def closed_form(beta):
    w = REF * np.exp(R_STAR / beta)
    return w / w.sum()


def test_recovers_the_closed_form_policy():
    for beta in (1.0, 0.5):
        theta, _ = run(beta)
        pi = softmax(theta)
        want = closed_form(beta)
        assert np.allclose(pi, want, atol=1e-3), f"beta={beta}: expected pi {want}, got {pi}"


def test_recovers_the_preference_order():
    theta, _ = run(0.5)
    pi = softmax(theta)
    ratio = pi / REF
    assert list(np.argsort(-ratio)) == [2, 1, 0], f"expected the order [2, 1, 0] by pi/pi_ref, got {np.argsort(-ratio)} from ratios {ratio}"


def test_implicit_reward_differences_match_the_true_reward():
    theta, _ = run(0.5)
    rhat = 0.5 * np.log(softmax(theta) / REF)
    d = np.diff(rhat)
    assert np.allclose(d, [1.0, 1.0], atol=1e-2), f"expected reward gaps [1, 1], got {d}"


def test_loss_history():
    _, losses = run(0.5, steps=200)
    assert len(losses) == 200, f"expected 200 losses, got {len(losses)}"
    assert abs(losses[0] - np.log(2)) < 1e-9, f"the first loss is at pi = pi_ref, expected log 2 = {np.log(2):.4f}, got {losses[0]}"
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])), "the loss must not increase with a small learning rate"
    assert losses[-1] < 0.6, f"expected the loss to fall below 0.6 after 200 steps, got {losses[-1]}"


def test_reference_is_not_modified():
    ref_logits = np.log(REF)
    before = ref_logits.copy()
    train_dpo(ref_logits, C, 0.5, 32.0, 50)
    assert np.array_equal(ref_logits, before), f"ref_logits changed from {before} to {ref_logits}; copy it before the update"


def test_zero_steps_returns_the_reference():
    theta, losses = train_dpo(np.log(REF), C, 0.5, 32.0, 0)
    assert np.allclose(theta, np.log(REF)), f"with 0 steps expected {np.log(REF)}, got {theta}"
    assert len(losses) == 0, f"expected no losses, got {losses}"


def test_learning_rate_sets_the_step_size():
    ref_logits = np.log(REF)
    a, _ = train_dpo(ref_logits, C, 0.5, 2.0, 1)
    b, _ = train_dpo(ref_logits, C, 0.5, 4.0, 1)
    da, db = a - ref_logits, b - ref_logits
    assert np.linalg.norm(da) > 0, "one step must move theta"
    assert np.allclose(db, 2 * da), f"doubling the learning rate must double the first step: {da} vs {db}"
