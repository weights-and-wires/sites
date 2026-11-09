import numpy as np


def epsilon_at(frame, eps_start=1.0, eps_end=0.1, anneal_frames=1_000_000):
    """Linear anneal from eps_start to eps_end over anneal_frames, then constant."""
    raise NotImplementedError


def select_action(q_values, epsilon, rng):
    """Epsilon-greedy over a 1-D array of Q-values. rng is a numpy Generator."""
    raise NotImplementedError
