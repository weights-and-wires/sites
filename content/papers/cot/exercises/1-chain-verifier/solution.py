import re

NUM = r"-?\d+(?:\.\d+)?"
EQUATION = re.compile(rf"(?<![\d.])({NUM})\s*([+\-*/])\s*({NUM})\s*=\s*({NUM})")
ANSWER = re.compile(rf"[Tt]he answer is\s*\$?({NUM})")


def _holds(a, op, b, c_text):
    """True when `a op b` equals the written result, to the decimals written."""
    a, b, c = float(a), float(b), float(c_text)
    if op == "+":
        value = a + b
    elif op == "-":
        value = a - b
    elif op == "*":
        value = a * b
    else:
        if b == 0:
            return False
        value = a / b
    decimals = len(c_text.split(".")[1]) if "." in c_text else 0
    tol = 1e-9 if decimals == 0 else 0.5 * 10 ** (-decimals) + 1e-9
    return abs(value - c) <= tol


def verify_chain(trace):
    """Return (answer, n_steps, bad_steps) for one generated chain of thought."""
    steps = 0
    bad = []
    for m in EQUATION.finditer(trace):
        steps += 1
        if not _holds(m.group(1), m.group(2), m.group(3), m.group(4)):
            bad.append(m.group(0))
    found = ANSWER.findall(trace)
    answer = float(found[-1]) if found else None
    return answer, steps, bad


def score_traces(traces, golds):
    """Return answer_acc, chain_valid, and lucky_correct for a list of traces."""
    if len(traces) != len(golds):
        raise ValueError(f"got {len(traces)} traces and {len(golds)} gold answers")
    n = len(traces)
    correct = valid = lucky = 0
    for trace, gold in zip(traces, golds):
        answer, steps, bad = verify_chain(trace)
        right = answer is not None and abs(answer - gold) < 1e-9
        sound = steps > 0 and not bad
        correct += right
        valid += sound
        lucky += right and not sound
    return {
        "answer_acc": correct / n if n else 0.0,
        "chain_valid": valid / n if n else 0.0,
        "lucky_correct": lucky / correct if correct else 0.0,
    }
