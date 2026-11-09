import numpy as np
from submission import train


def sv(M):
    return np.linalg.svd(M, compute_uv=False)


def test_losses_have_one_entry_per_step():
    out = train(True, steps=7)
    assert len(out["losses"]) == 7, f"expected 7 losses, got {len(out['losses'])}"
    assert set(out["deltas"]) == {"W_in", "W_h", "W_out"}, f"expected deltas for W_in, W_h, W_out, got {sorted(out['deltas'])}"


def test_muon_run_learns():
    losses = train(True)["losses"]
    assert losses[-1] < 0.05 * losses[0], f"final loss expected below {0.05 * losses[0]:.4f}, got {losses[-1]:.4f}"


def test_adamw_run_learns():
    losses = train(False)["losses"]
    assert losses[-1] < 0.05 * losses[0], f"final loss expected below {0.05 * losses[0]:.4f}, got {losses[-1]:.4f}"


def test_muon_first_step_is_orthogonalized():
    d = train(True, steps=1)["deltas"]["W_h"] / 0.02
    s = sv(d)
    assert 0.6 < s[0] < 1.3, f"top singular value of the first W_h step divided by lr_muon expected in (0.6, 1.3), got {s[0]:.3f}"


def test_adamw_path_for_hidden_matrix_is_sign_like():
    # The first AdamW step has magnitude lr_adam in every entry whose gradient is not 0.
    d = train(False, steps=1)["deltas"]["W_h"]
    assert np.isclose(np.abs(d).max(), 0.01, atol=1e-6), f"expected largest first-step entry 0.01, got {np.abs(d).max()}"
    assert sv(d)[0] / 0.01 > 3, f"an AdamW step is not orthogonalized; expected a large top singular value, got {sv(d)[0] / 0.01:.2f}"


def test_input_and_output_matrices_use_adamw_in_both_modes():
    for use_muon in (True, False):
        deltas = train(use_muon, steps=1)["deltas"]
        for name in ("W_in", "W_out"):
            m = np.abs(deltas[name]).max()
            assert np.isclose(m, 0.01, atol=1e-6), f"use_muon={use_muon}: {name} first step must be an AdamW step with entries up to 0.01, got {m}"


def test_momentum_buffer_persists_across_steps():
    # Reference losses at index 19 (the 20th step) come from a correct implementation.
    got_m = train(True, steps=20)["losses"][19]
    got_a = train(False, steps=20)["losses"][19]
    assert np.isclose(got_m, 0.035265, rtol=1e-3), f"Muon run: expected loss 0.035265 before step 20, got {got_m:.6f}"
    assert np.isclose(got_a, 0.027259, rtol=1e-3), f"AdamW run: expected loss 0.027259 before step 20, got {got_a:.6f}"
