"""Process labels, outcome labels, and synthetic labels (Sections 2.5, 2.6, Appendix H).

Run: python3 scripts/step_labels.py
"""


def process_labels(step_ok):
    """Labels up to and including the first incorrect step (Section 2.6)."""
    labels = []
    for ok in step_ok:
        labels.append(int(ok))
        if not ok:
            break
    return labels


def outcome_labels(n_steps, final_correct):
    """The ORM target: the final-answer label, repeated for every position."""
    return [int(final_correct)] * n_steps


# (step correctness, final answer correct)
solutions = {
    "wrong, error at step 3": ([True, True, False, True, False], False),
    "all steps correct": ([True, True, True, True], True),
    "right answer, flawed step": ([True, False, True], True),
}
for name, (ok, final) in solutions.items():
    print(f"{name:26s} step_ok={[int(x) for x in ok]} final_correct={final}")
    print(f"{'':26s} process={process_labels(ok)}  outcome={outcome_labels(len(ok), final)}")

print()
print("Why a negative outcome label is weak evidence (Section 6.1)")
for per_step_error in (0.05, 0.10, 0.20):
    for n_steps in (5, 10, 20):
        ok = (1 - per_step_error) ** n_steps
        print(f"error/step={per_step_error:.2f} steps={n_steps:2d} P(all steps right)={ok:.4f}")

print()
print("Synthetic labels from PRM_large (Appendix H): incorrect if P(negative) > 0.20")
p_neg = [0.05, 0.10, 0.20, 0.25, 0.01]
step_ok = [p <= 0.20 for p in p_neg]
print(f"p_neg={p_neg}")
print(f"step_ok={[int(x) for x in step_ok]}")
print(f"process labels={process_labels(step_ok)}  outcome labels={outcome_labels(len(step_ok), all(step_ok))}")
