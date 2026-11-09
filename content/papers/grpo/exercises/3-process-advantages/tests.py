import numpy as np
from submission import process_advantages

STEPS = [np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])]


def test_page_example():
    got = process_advantages(STEPS)
    want = [[2.1213, 0.7071, -0.7071], [0.7071, -0.7071], [-2.8284, -2.1213, -1.4142, -0.7071]]
    assert len(got) == 3, f"expected 3 arrays, got {len(got)}"
    for i in range(3):
        assert np.allclose(got[i], want[i], atol=1e-3), f"output {i}: expected {want[i]}, got {np.asarray(got[i])}"


def test_shapes_follow_the_step_counts():
    got = process_advantages(STEPS)
    assert [len(g) for g in got] == [3, 2, 4], f"expected lengths [3, 2, 4], got {[len(g) for g in got]}"


def test_last_step_advantage_is_its_own_normalized_reward():
    got = process_advantages(STEPS)
    flat = np.concatenate(STEPS)
    for i, s in enumerate(STEPS):
        want = (s[-1] - flat.mean()) / flat.std()
        assert abs(got[i][-1] - want) < 1e-4, f"output {i}: the last step has no later steps, expected {want:.4f}, got {got[i][-1]:.4f}"


def test_sum_runs_over_later_steps_not_earlier():
    s = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    got = process_advantages(s)
    # Only output 0 step 0 is positive. A reverse cumulative sum gives a different
    # first advantage than a forward one: here the first step collects every later term.
    norm0 = (s[0] - 1 / 6) / np.sqrt(1 / 6 - 1 / 36)
    assert abs(got[0][0] - norm0.sum()) < 1e-3, f"step 0 must sum all steps: expected {norm0.sum():.4f}, got {got[0][0]:.4f}"
    assert abs(got[0][2] - norm0[2]) < 1e-3, f"step 2 must keep only itself: expected {norm0[2]:.4f}, got {got[0][2]:.4f}"


def test_normalizes_over_the_whole_group():
    got = process_advantages(STEPS)
    # Output 3 has only zero rewards. Per-output normalization would give it 0 advantage everywhere.
    assert got[2][-1] < -0.5, f"an all-bad output must get negative advantages, got {np.asarray(got[2])}"
    first_steps = [g[0] for g in got]
    assert first_steps[0] > first_steps[1] > first_steps[2], f"better outputs must rank higher at step 0, got {first_steps}"


def test_one_step_per_output_matches_outcome_supervision():
    r = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    got = np.array([g[0] for g in process_advantages([np.array([x]) for x in r])])
    want = (r - r.mean()) / r.std()
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"
