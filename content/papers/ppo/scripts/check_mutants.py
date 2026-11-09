#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/ppo/scripts/check_mutants.py
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
HARNESS = ROOT / "static/runner/harness.py"
EXERCISES = pathlib.Path(__file__).resolve().parents[1] / "exercises"

CLIP = "np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)"

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-clipped-surrogate", "no min: only the clipped term", [(CLIP, "np.clip(ratio, 1 - eps, 1 + eps) * adv")]),
    ("1-clipped-surrogate", "max instead of min (optimistic)", [(CLIP, "np.maximum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)")]),
    ("1-clipped-surrogate", "no clipping at all (L_CPI)", [(CLIP, "ratio * adv")]),
    ("1-clipped-surrogate", "ratio taken as logp difference, not exp", [("ratio = np.exp(np.asarray(logp_new) - np.asarray(logp_old))\n    adv = np.asarray(adv)\n    return", "ratio = np.asarray(logp_new) - np.asarray(logp_old) + 1\n    adv = np.asarray(adv)\n    return")]),
    ("1-clipped-surrogate", "sum instead of mean", [("return float(np.mean(np.minimum", "return float(np.sum(np.minimum")]),
    ("1-clipped-surrogate", "gradient forgets the ratio factor", [("np.where(active, unclipped, 0.0)", "np.where(active, adv, 0.0)")]),
    ("1-clipped-surrogate", "gradient is never zero (ignores clipping)", [("np.where(active, unclipped, 0.0)", "unclipped")]),
    ("1-clipped-surrogate", "gradient zero whenever r leaves the range", [("active = unclipped <= clipped", "active = np.abs(ratio - 1) <= eps")]),
    ("1-clipped-surrogate", "gradient not divided by N", [("/ len(ratio)", "")]),
    ("2-adaptive-kl", "KL direction reversed", [("p_old * (np.log(p_old) - np.log(p_new))", "p_new * (np.log(p_new) - np.log(p_old))")]),
    ("2-adaptive-kl", "KL summed over rows", [("return float(np.mean(np.sum(p_old", "return float(np.sum(np.sum(p_old")]),
    ("2-adaptive-kl", "penalty added, not subtracted", [("- beta * kl_categorical(p_old, p_new)", "+ beta * kl_categorical(p_old, p_new)")]),
    ("2-adaptive-kl", "beta moves the wrong way", [("return beta / 2\n    if d > d_targ * 1.5:\n        return beta * 2", "return beta * 2\n    if d > d_targ * 1.5:\n        return beta / 2")]),
    ("2-adaptive-kl", "lower threshold d_targ * 1.5 instead of / 1.5", [("d < d_targ / 1.5", "d < d_targ * 1.5")]),
    ("2-adaptive-kl", "factor 1.5 swapped for 2 in the thresholds", [("d_targ / 1.5", "d_targ / 2"), ("d_targ * 1.5:\n        return beta * 2", "d_targ * 2:\n        return beta * 2")]),
    ("3-combined-objective", "value loss added instead of subtracted", [("- c1 * l_vf", "+ c1 * l_vf")]),
    ("3-combined-objective", "entropy subtracted", [("+ c2 * entropy(probs)", "- c2 * entropy(probs)")]),
    ("3-combined-objective", "value error not squared", [("(values - returns) ** 2", "np.abs(values - returns)")]),
    ("3-combined-objective", "entropy of the old policy", [("c2 * entropy(probs)", "c2 * entropy(probs_old)")]),
    ("3-combined-objective", "entropy in bits (log2)", [("np.log(probs)", "np.log2(probs)")]),
    ("3-combined-objective", "ratio uses the first action of every row", [("probs[idx, actions] / probs_old[idx, actions]", "probs[idx, 0] / probs_old[idx, 0]")]),
    ("3-combined-objective", "loss keeps the objective's sign", [("return -ppo_objective(", "return ppo_objective(")]),
    ("4-bandit-epochs", "old policy refreshed every epoch (ratio always 1)", [("        pi = softmax(theta)\n        g = ", "        pi = softmax(theta)\n        logp_old = np.log(pi)[actions]\n        g = ")]),
    ("4-bandit-epochs", "gradient w.r.t. logits uses -pi only (no one-hot)", [("g @ (onehot - pi)", "g @ (0 * onehot - pi)")]),
    ("4-bandit-epochs", "descent instead of ascent", [("theta = theta + lr *", "theta = theta - lr *")]),
    ("4-bandit-epochs", "clipping disabled", [("clip_grad_logp(np.log(pi)[actions], logp_old, adv, eps)", "clip_grad_logp(np.log(pi)[actions], logp_old, adv, None)")]),
    ("4-bandit-epochs", "only one epoch runs", [("for _ in range(epochs):", "for _ in range(min(epochs, 1)):")]),
    ("4-bandit-epochs", "mutates theta_old in place", [("theta = np.array(theta_old, dtype=float)", "theta = theta_old"), ("theta = theta + lr * (g @ (onehot - pi))", "theta += lr * (g @ (onehot - pi))")]),
]


def run_tests(ex, source):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "submission.py").write_text(source)
        shutil.copy(ex / "tests.py", tmp / "tests.py")
        shutil.copy(HARNESS, tmp / "harness.py")
        proc = subprocess.run([sys.executable, "harness.py", "."], cwd=tmp, capture_output=True, text=True, timeout=120)
        lines = proc.stdout.strip().splitlines()
        if not lines:
            return None, proc.stderr[-400:]
        return json.loads(lines[-1]), ""


def main():
    survivors = 0
    ran = 0
    for folder, name, edits in MUTANTS:
        ex = EXERCISES / folder
        solution = (ex / "solution.py").read_text()
        base, err = run_tests(ex, solution)
        if base is None or not all(r["ok"] for r in base):
            print(f"ERROR {folder}: the unmodified solution does not pass: {err}")
            survivors += 1
            continue
        mutated = solution
        for old, new in edits:
            if old not in mutated:
                print(f"ERROR {folder}: {name}: pattern not found: {old!r}")
                sys.exit(2)
            mutated = mutated.replace(old, new)
        results, err = run_tests(ex, mutated)
        ran += 1
        if results is None:
            print(f"ERROR {folder}: {name}: harness crashed: {err}")
            survivors += 1
            continue
        failed = [r["name"] for r in results if not r["ok"]]
        if failed:
            print(f"CAUGHT   {folder}: {name}\n         failing: {', '.join(failed)}")
        else:
            print(f"SURVIVED {folder}: {name}\n         no test failed")
            survivors += 1
    print(f"\n{ran - survivors} of {ran} mutants caught, {survivors} survived")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
