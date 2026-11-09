import numpy as np
from submission import discriminator_loss, generator_adv_loss, generator_loss

REAL = [np.array([1.5, 0.2, 2.0, -0.3])]
FAKE = [np.array([-0.5, 0.8, -2.0, 1.2])]


def test_discriminator_loss_page_example():
    got = discriminator_loss(REAL, FAKE)
    assert np.isclose(got, 1.65), f"expected 1.65 (0.525 + 1.125), got {got}"


def test_generator_loss_page_example():
    got = generator_adv_loss(FAKE)
    assert np.isclose(got, 1.175), f"expected 1.175, got {got}"


def test_hinge_clips_at_the_margin():
    got = generator_adv_loss([np.array([1.0, 2.0, 5.0])])
    assert np.isclose(got, 0.0), f"logits at or above 1 must cost the generator nothing; expected 0, got {got}"
    got = discriminator_loss([np.array([1.0, 3.0])], [np.array([-1.0, -4.0])])
    assert np.isclose(got, 0.0), f"a discriminator beyond the margin on both sides has zero loss; expected 0, got {got}"


def test_real_and_fake_terms_have_opposite_signs():
    only_real = discriminator_loss([np.array([-1.0])], [np.array([-5.0])])
    assert np.isclose(only_real, 2.0), f"real logit -1 costs max(0, 1 - (-1)) = 2 and fake logit -5 costs 0; expected 2, got {only_real}"
    only_fake = discriminator_loss([np.array([5.0])], [np.array([0.5])])
    assert np.isclose(only_fake, 1.5), f"fake logit 0.5 costs max(0, 1 + 0.5) = 1.5; expected 1.5, got {only_fake}"


def test_average_over_time_before_discriminators():
    # Discriminator 0 has 2 logits, discriminator 1 has 8. Per-k means are 1.0 and 1.5.
    fake = [np.array([0.0, 0.0]), np.array([1.0] * 6 + [-1.0] * 2)]
    got = generator_adv_loss([np.array([0.0, 0.0]), np.array([-0.5] * 6 + [1.5] * 2)])
    want = np.mean([np.mean([1.0, 1.0]), np.mean([1.5] * 6 + [0.0] * 2)])
    assert np.isclose(got, want), f"expected the mean of per-discriminator means {want}, got {got}"
    got_d = discriminator_loss([np.array([1.0])], fake)
    assert np.isclose(got_d, 1.25), f"fake term must be (1.0 + 1.5) / 2 = 1.25, not the flat mean 1.4; got {got_d}"


def test_generator_loss_weights():
    got = generator_loss(1.0, 0.02, 3.0)
    assert np.isclose(got, 1.0 + 100 * 0.02 + 3.0), f"expected 1 + 100 * 0.02 + 3 = 6, got {got}"
    got = generator_loss(1.0, 1.0, 1.0, lam_adv=2.0, lam_feat=3.0, lam_rec=4.0)
    assert np.isclose(got, 9.0), f"custom weights 2, 3, 4 on losses 1, 1, 1: expected 9, got {got}"
