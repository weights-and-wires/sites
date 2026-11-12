"""Score two hypotheses with the beam-search objective Q(y) (Section 3.8, Equation 12).

The log probabilities below are made up for illustration. The paper tunes alpha and beta on a
development set and does not report their values.

Run: python3 scripts/lm_rescoring_example.py
"""

hyps = {
    "recognize speech": {"log_ctc": -4.0, "log_lm": -6.0, "words": 2},
    "wreck a nice beach": {"log_ctc": -3.5, "log_lm": -14.0, "words": 4},
}


def q(h, alpha, beta):
    return h["log_ctc"] + alpha * h["log_lm"] + beta * h["words"]


for alpha, beta in ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.5, 2.0)):
    scores = {name: q(h, alpha, beta) for name, h in hyps.items()}
    best = max(scores, key=scores.get)
    shown = ", ".join(f"{name!r}: {score:+.2f}" for name, score in scores.items())
    print(f"alpha={alpha}, beta={beta}: {shown}  -> best {best!r}")
