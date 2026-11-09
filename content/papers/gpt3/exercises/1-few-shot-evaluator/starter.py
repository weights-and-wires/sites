import numpy as np


def format_prompt(demos, query, description=None):
    """Return the few-shot prompt text.

    demos: list of (question, answer) strings. query: the question to answer.
    description: optional task description placed before the demonstrations.
    """
    raise NotImplementedError


def evaluate(model, dataset, pool, k, seed=0):
    """Exact-match accuracy of model(prompt) -> str over dataset, with k demos per query."""
    raise NotImplementedError
