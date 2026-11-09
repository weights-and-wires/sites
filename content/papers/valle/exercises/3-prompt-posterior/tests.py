import numpy as np
from submission import speaker_posterior, next_token_dist, continue_greedy

V, K = 6, 3
trans = np.full((K, V, V), 0.3 / (V - 1))
for k in range(K):
    for v in range(V):
        trans[k, v, (v + k + 1) % V] = 0.7
prior = np.ones(K) / K


def test_one_token_gives_the_prior():
    got = speaker_posterior([3], trans, prior)
    assert np.allclose(got, prior), f"no transition seen, so expected the prior {prior}, got {got}"


def test_two_token_prompt_by_hand():
    # Speaker likelihoods for 0 -> 2: 0.06, 0.7, 0.06.
    got = speaker_posterior([0, 2], trans, prior)
    want = np.array([0.06, 0.7, 0.06]) / 0.82
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_prior_matters():
    skew = np.array([0.8, 0.1, 0.1])
    got = speaker_posterior([0, 2], trans, skew)
    want = np.array([0.8 * 0.06, 0.1 * 0.7, 0.1 * 0.06])
    want = want / want.sum()
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_long_prompt_does_not_underflow():
    # Speaker 1's path with one slip every 500 tokens. Raw products of 0.7 and 0.06
    # underflow to exactly 0 for every speaker, so a non-log implementation divides 0 by 0.
    prompt = [(2 * t + (t // 500)) % V for t in range(3000)]
    got = speaker_posterior(prompt, trans, prior)
    assert np.all(np.isfinite(got)), f"posterior must be finite for a 3000-token prompt, got {got}"
    assert np.allclose(got, [0, 1, 0], atol=1e-6), f"expected [0, 1, 0], got {got}"


def test_transition_direction():
    # 0 -> 1 is a step of 1 (speaker 0). The reverse, 1 -> 0, is a step of 5, which no speaker prefers.
    got = speaker_posterior([0, 1], trans, prior)
    assert int(np.argmax(got)) == 0 and got[0] > 0.8, f"0 -> 1 must favor speaker 0 (> 0.8), got {got}"
    got = speaker_posterior([1, 0], trans, prior)
    assert np.allclose(got, prior), f"1 -> 0 matches no speaker's favored step, so expected the prior, got {got}"


def test_next_token_mixture_by_hand():
    got = next_token_dist([0, 2], trans, prior)
    want = np.array([0.06, 0.06, 0.06, 0.1068, 0.6063, 0.1068])
    assert np.allclose(got, want, atol=1e-3), f"expected {want}, got {got}"
    assert np.isclose(got.sum(), 1), f"must sum to 1, got {got.sum()}"


def test_empty_context_is_the_plain_mixture():
    got = next_token_dist([0], trans, prior)
    want = prior @ trans[:, 0, :]
    assert np.allclose(got, want), f"expected the prior mixture {want}, got {got}"


def test_continuation_follows_the_prompted_speaker():
    got = continue_greedy([0, 2], 5, trans, prior)
    assert got == [0, 2, 4, 0, 2, 4, 0], f"expected [0, 2, 4, 0, 2, 4, 0], got {got}"
    got = continue_greedy([0, 3], 4, trans, prior)
    assert got == [0, 3, 0, 3, 0, 3], f"a prompt that steps by 3 must continue by 3; got {got}"


def test_one_token_prompt_is_ambiguous():
    got = continue_greedy([0], 1, trans, prior)
    assert len(got) == 2 and got[0] == 0, f"expected the prefix to be kept, got {got}"
