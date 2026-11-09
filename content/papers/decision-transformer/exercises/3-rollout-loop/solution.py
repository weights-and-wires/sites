import numpy as np


def rollout(policy, env_step, s0, target_return, K, max_steps):
    """Run one episode with return-to-go conditioning.

    policy(R_ctx, s_ctx, a_ctx, t_ctx) -> action, with arrays of length n <= K
    (a_ctx has length n - 1).
    env_step(s, a) -> (next_state, reward, done).
    Returns (actions, total_reward).
    """
    R, s, a, t = [float(target_return)], [s0], [], [1]
    total = 0.0
    for _ in range(max_steps):
        n = min(K, len(R))
        R_ctx = np.array(R[-n:])
        s_ctx = np.array(s[-n:])
        t_ctx = np.array(t[-n:])
        a_ctx = np.array(a[len(a) - (n - 1):])
        action = policy(R_ctx, s_ctx, a_ctx, t_ctx)
        s_next, r, done = env_step(s[-1], action)
        a.append(action)
        total += r
        if done:
            break
        R.append(R[-1] - r)
        s.append(s_next)
        t.append(t[-1] + 1)
    return a, total
