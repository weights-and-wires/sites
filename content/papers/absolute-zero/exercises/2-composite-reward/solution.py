def composite_reward(r_role, well_formed, passed):
    """Return the composite reward of Equation 6 as a float.

    r_role: the role reward (solver 0 or 1, proposer in [0, 1]).
    well_formed: True if the response has the required format.
    passed: True if the answer is correct (solver) or the task is valid (proposer).
    """
    if not well_formed:
        return -1.0
    if passed:
        return float(r_role)
    return -0.5
