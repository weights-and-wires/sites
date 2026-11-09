import numpy as np
from submission import rollout


def line_env(rewards):
    """State is the step index; reward is rewards[state]; done after the last reward."""
    def env_step(s, a):
        return s + 1, rewards[s], s + 1 >= len(rewards)
    return env_step


def recorder(calls, action=0):
    def policy(R, s, a, t):
        calls.append((R.tolist(), s.tolist(), a.tolist(), t.tolist()))
        return action
    return policy


def test_return_to_go_is_decremented_by_each_reward():
    calls = []
    rollout(recorder(calls), line_env([1, 0, 2, -1, 5]), 0, 10.0, K=10, max_steps=5)
    last_R = [c[0][-1] for c in calls]
    assert last_R == [10, 9, 9, 7, 8], f"expected current return-to-go [10, 9, 9, 7, 8], got {last_R}"


def test_first_call_has_a_context_of_one():
    calls = []
    rollout(recorder(calls), line_env([1, 1, 1]), 0, 3.0, K=4, max_steps=3)
    assert calls[0] == ([3.0], [0], [], [1]), f"expected ([3.0], [0], [], [1]), got {calls[0]}"


def test_context_is_capped_at_K_and_slides():
    calls = []
    rollout(recorder(calls, action=7), line_env([1] * 6), 0, 6.0, K=3, max_steps=6)
    sizes = [len(c[0]) for c in calls]
    assert sizes == [1, 2, 3, 3, 3, 3], f"expected context lengths [1, 2, 3, 3, 3, 3], got {sizes}"
    assert calls[4][1] == [2, 3, 4], f"expected the states [2, 3, 4] at step 4, got {calls[4][1]}"
    assert calls[4][0] == [4, 3, 2], f"expected returns-to-go [4, 3, 2] at step 4, got {calls[4][0]}"


def test_timesteps_are_absolute_and_start_at_one():
    calls = []
    rollout(recorder(calls), line_env([0] * 6), 0, 0.0, K=3, max_steps=6)
    got = [c[3] for c in calls]
    want = [[1], [1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    assert got == want, f"expected timesteps {want}, got {got}"


def test_action_context_is_one_shorter_and_aligned():
    calls = []
    actions = iter(range(100, 110))

    def policy(R, s, a, t):
        calls.append((len(R), a.tolist()))
        return next(actions)

    rollout(policy, line_env([0] * 5), 0, 0.0, K=3, max_steps=5)
    want = [(1, []), (2, [100]), (3, [100, 101]), (3, [101, 102]), (3, [102, 103])]
    assert calls == want, f"expected (context length, past actions) {want}, got {calls}"


def test_K_equals_one_has_no_actions():
    calls = []
    rollout(recorder(calls), line_env([1] * 4), 0, 4.0, K=1, max_steps=4)
    assert all(c[2] == [] and len(c[0]) == 1 for c in calls), f"with K = 1 every call should see one timestep and no actions, got {calls}"


def test_returns_actions_and_total_reward():
    actions, total = rollout(recorder([], action=3), line_env([1, 0, 2, -1]), 0, 2.0, K=2, max_steps=10)
    assert actions == [3, 3, 3, 3], f"the episode ends at done; expected four actions [3, 3, 3, 3], got {actions}"
    assert total == 2.0, f"expected total reward 2.0, got {total}"


def test_max_steps_stops_the_episode():
    actions, total = rollout(recorder([]), line_env([1] * 10), 0, 10.0, K=3, max_steps=4)
    assert len(actions) == 4, f"expected 4 actions with max_steps=4, got {len(actions)}"
    assert total == 4.0, f"expected total reward 4.0, got {total}"
