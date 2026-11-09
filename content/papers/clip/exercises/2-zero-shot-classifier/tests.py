import numpy as np
from submission import build_classifier, zero_shot_probs


def at(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


# The example from the page: two classes, two prompts, embeddings given as angles.
PROMPTS = np.array([[at(60), at(5)], [at(20), at(100)]])
IMAGE = at(30)[None, :]


def test_single_prompt_is_the_normalized_embedding():
    emb = np.array([[[3.0, 4.0], [0.0, 2.0]]])
    out = build_classifier(emb)
    assert out.shape == (2, 2), f"expected shape (2, 2), got {out.shape}"
    assert np.allclose(out, [[0.6, 0.8], [0.0, 1.0]]), f"expected [[0.6, 0.8], [0, 1]], got {out}"


def test_ensemble_rows_are_unit_length():
    rng = np.random.default_rng(0)
    W = build_classifier(rng.normal(size=(5, 7, 4)))
    norms = np.linalg.norm(W, axis=1)
    assert W.shape == (7, 4), f"expected shape (7, 4), got {W.shape}"
    assert np.allclose(norms, 1), f"every row must be renormalized after the average; norms are {norms}"


def test_ensemble_values_from_the_page():
    W = build_classifier(PROMPTS)
    angles = np.rad2deg(np.arctan2(W[:, 1], W[:, 0]))
    assert np.allclose(angles, [40.0, 52.5], atol=1e-6), f"expected angles [40, 52.5] degrees, got {angles}"


def test_each_prompt_is_normalized_before_averaging():
    scaled = PROMPTS.copy()
    scaled[1] *= 7.0  # prompt B gets longer vectors but points the same way
    W = build_classifier(scaled)
    angles = np.rad2deg(np.arctan2(W[:, 1], W[:, 0]))
    assert np.allclose(angles, [40.0, 52.5], atol=1e-6), f"a longer prompt embedding must not outvote the other; expected [40, 52.5], got {angles}"


def test_average_runs_over_prompts_not_classes():
    emb = np.zeros((2, 3, 2))
    emb[0] = [[1, 0], [1, 0], [1, 0]]
    emb[1] = [[0, 1], [0, 1], [0, 1]]
    W = build_classifier(emb)
    assert np.allclose(W, np.tile([np.sqrt(0.5), np.sqrt(0.5)], (3, 1))), f"expected every class at 45 degrees, got {W}"


def test_probabilities_sum_to_one_over_classes():
    rng = np.random.default_rng(1)
    W = build_classifier(rng.normal(size=(3, 4, 6)))
    p = zero_shot_probs(rng.normal(size=(5, 6)), W, np.log(100.0))
    assert p.shape == (5, 4), f"expected shape (5, 4), got {p.shape}"
    assert np.allclose(p.sum(axis=1), 1), f"each image's probabilities must sum to 1 over the classes, got {p.sum(axis=1)}"


def test_ensemble_fixes_what_one_prompt_gets_wrong():
    t = np.log(100.0)
    a = zero_shot_probs(IMAGE, build_classifier(PROMPTS[:1]), t)[0]
    b = zero_shot_probs(IMAGE, build_classifier(PROMPTS[1:]), t)[0]
    both = zero_shot_probs(IMAGE, build_classifier(PROMPTS), t)[0]
    assert np.allclose(a, [0.0175, 0.9825], atol=1e-3), f"prompt A alone: expected [0.0175, 0.9825], got {a}"
    assert np.allclose(b, [1.0, 0.0], atol=1e-3), f"prompt B alone: expected [1, 0], got {b}"
    assert np.allclose(both, [0.9977, 0.0023], atol=1e-3), f"ensemble: expected [0.9977, 0.0023], got {both}"


def test_scale_is_exp_of_t():
    W = build_classifier(PROMPTS)
    p0 = zero_shot_probs(IMAGE, W, 0.0)[0]
    cos = (W @ IMAGE[0])
    want = np.exp(cos) / np.exp(cos).sum()
    assert np.allclose(p0, want), f"t=0 means a logit scale of 1; expected {want}, got {p0}"


def test_image_embeddings_are_normalized():
    W = build_classifier(PROMPTS)
    small = zero_shot_probs(IMAGE, W, np.log(100.0))
    big = zero_shot_probs(IMAGE * 9.0, W, np.log(100.0))
    assert np.allclose(small, big), f"only the direction of the image embedding may matter; got {small} and {big}"


def test_stable_for_large_logits():
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    p = zero_shot_probs(np.array([[1.0, 0.0]]), W, np.log(5000.0))
    assert np.all(np.isfinite(p)), f"probabilities are {p} at scale 5000; subtract the row maximum"
    assert np.allclose(p, [[1.0, 0.0]]), f"expected [[1, 0]], got {p}"
