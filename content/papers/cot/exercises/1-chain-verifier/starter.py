import re

NUM = r"-?\d+(?:\.\d+)?"
EQUATION = re.compile(rf"(?<![\d.])({NUM})\s*([+\-*/])\s*({NUM})\s*=\s*({NUM})")
ANSWER = re.compile(rf"[Tt]he answer is\s*\$?({NUM})")


def verify_chain(trace):
    """Return (answer, n_steps, bad_steps) for one generated chain of thought.

    answer: the number after the last "The answer is" (a float), or None.
    n_steps: how many single-operator equations `a op b = c` the trace contains.
    bad_steps: the text of each equation that is arithmetically wrong.
    """
    raise NotImplementedError


def score_traces(traces, golds):
    """Return a dict with answer_acc, chain_valid, and lucky_correct.

    answer_acc: share of traces whose answer equals the gold answer.
    chain_valid: share of traces with at least one equation and no wrong equation.
    lucky_correct: among traces with the right answer, the share whose chain is not valid.
    Raise ValueError when the two lists differ in length.
    """
    raise NotImplementedError
