"""Reward and verification helpers for the Absolute Zero page (stdlib + NumPy)."""

import numpy as np


def learnability_reward(solver_rewards):
    """Equation 4: 0 if the mean solver reward is 0, else 1 - mean."""
    mean = float(np.mean(np.asarray(solver_rewards, dtype=float)))
    if mean == 0.0:
        return 0.0
    return 1.0 - mean


def composite_reward(r_role, well_formed, passed):
    """Equation 6: r_role, -0.5, or -1."""
    if not well_formed:
        return -1.0
    if passed:
        return float(r_role)
    return -0.5


def task_relative_advantages(rewards, tasks, roles, eps=1e-6):
    """Equation 8: normalize within each (task, role) group."""
    rewards = np.asarray(rewards, dtype=float)
    out = np.zeros_like(rewards)
    keys = list(zip(tasks, roles))
    for key in set(keys):
        idx = np.array([k == key for k in keys])
        group = rewards[idx]
        out[idx] = (group - group.mean()) / (group.std() + eps)
    return out


def run_program(src, x):
    """Run `def f(x)` from source and return f(x)."""
    env = {}
    exec(src, {"__builtins__": {"range": range, "sum": sum, "len": len, "abs": abs, "min": min, "max": max}}, env)
    return env["f"](x)


def check_answer(kind, program_src, task_input, task_output, answer):
    """Verify a solver answer for one triplet (p, i, o)."""
    if kind == "deduction":
        return answer == task_output
    if kind == "abduction":
        try:
            return run_program(program_src, answer) == task_output
        except Exception:
            return False
    if kind == "induction":
        try:
            return all(run_program(answer, x) == y for x, y in task_input)
        except Exception:
            return False
    raise ValueError(kind)
