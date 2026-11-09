import numpy as np


def rollout(policy, env_step, s0, target_return, K, max_steps):
    """Run one episode with return-to-go conditioning.

    policy(R_ctx, s_ctx, a_ctx, t_ctx) -> action, with arrays of length n <= K
    (a_ctx has length n - 1).
    env_step(s, a) -> (next_state, reward, done).
    Returns (actions, total_reward).
    """
    raise NotImplementedError
