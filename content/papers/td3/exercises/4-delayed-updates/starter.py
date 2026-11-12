def actor_update_steps(total_steps, d):
    """Return the 1-based steps t in 1..total_steps where t is a multiple of d."""
    raise NotImplementedError


def target_gap(gap0, tau, total_steps, d):
    """Return the gap left after total_steps critic steps if targets move only on actor-update steps."""
    raise NotImplementedError
