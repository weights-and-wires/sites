import numpy as np

from submission import diversity_loss


def test_uniform_codebook_value():
    got = diversity_loss(np.zeros((5, 1, 4)))
    want = -np.log(4) / 4
    assert abs(got - want) < 1e-9, f"uniform V=4, G=1: expected -log(4)/4 = {want}, got {got}"


def test_paper_sized_codebook_uniform():
    got = diversity_loss(np.zeros((3, 2, 320)))
    want = -np.log(320) / 320
    assert abs(got - want) < 1e-9, f"G=2, V=320 uniform: expected {want}, got {got}"


def test_skewed_distribution_value():
    p = np.array([0.7, 0.1, 0.1, 0.1])
    logits = np.log(p)[None, None, :]
    want = float((p * np.log(p)).sum() / 4)
    got = diversity_loss(logits)
    assert abs(got - want) < 1e-9, f"p = [.7, .1, .1, .1]: expected {want:.4f}, got {got}"


def test_entropy_of_the_average_not_average_of_entropies():
    # Two steps, each almost certain, but about different entries: the average is uniform.
    logits = np.array([[[50.0, 0.0]], [[0.0, 50.0]]])
    got = diversity_loss(logits)
    want = -np.log(2) / 2
    assert abs(got - want) < 1e-6, f"confident steps on different entries: expected {want:.4f}, got {got}"
    same = np.array([[[50.0, 0.0]], [[50.0, 0.0]]])
    got2 = diversity_loss(same)
    assert abs(got2) < 1e-6, f"confident steps on the same entry: expected about 0, got {got2}"


def test_groups_are_summed_and_scaled_by_G_V():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, 3, 5))
    parts = [diversity_loss(logits[:, g:g + 1, :]) for g in range(3)]
    # Each single-group call divides by V; the full call divides by G * V.
    want = sum(parts) / 3
    got = diversity_loss(logits)
    assert abs(got - want) < 1e-12, f"expected the mean of the per-group values {want}, got {got}"


def test_lower_bound_and_ordering():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(8, 2, 6)) * 3
    got = diversity_loss(logits)
    low = -np.log(6) / 6
    assert low - 1e-12 <= got <= 0.0, f"L_d lies in [-log(V)/V, 0] = [{low:.4f}, 0], got {got}"
    assert got > diversity_loss(np.zeros((8, 2, 6))), "a skewed average must give a larger loss than a uniform one"


def test_stable_for_large_logits():
    got = diversity_loss(np.array([[[1000.0, 0.0, -1000.0]]]))
    assert np.isfinite(got), f"large logits must not overflow, got {got}"
    assert abs(got) < 1e-9, f"a one-hot average has entropy 0, expected 0.0, got {got}"
