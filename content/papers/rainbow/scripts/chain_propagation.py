"""How fast a terminal reward reaches the start of a chain.

A deterministic chain of 12 states: the agent moves right, and the last
transition pays reward 1. One sweep updates every state at once from the values of the previous sweep
(rewards are undiscounted for clarity). The script counts how many sweeps the
start of the chain needs before its value becomes nonzero, for n-step targets
with n = 1 and n = 3.

Run: python3 scripts/chain_propagation.py
"""

import numpy as np

L = 12  # states 0 .. L-1, reward 1 on leaving state L-1


def sweeps_until_start_learns(n):
    v = np.zeros(L + 1)  # v[L] is the terminal state, value 0
    for sweep in range(1, 10 * L):
        old = v.copy()
        for s in range(L):
            end = min(s + n, L)
            ret = 1.0 if s + n >= L else 0.0  # the reward arrives on the last step
            v[s] = ret + old[end]
        if v[0] > 0:
            return sweep
    return None


for n in (1, 3):
    print(f"n = {n}: the start of the {L}-state chain learns the reward after {sweeps_until_start_learns(n)} sweeps")
