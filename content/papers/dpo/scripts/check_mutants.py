#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/dpo/scripts/check_mutants.py 
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

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-closed-form-policy", "ignore the reference policy", [("log_weights = np.log(ref_probs) + r / beta", "log_weights = r / beta")]),
    ("1-closed-form-policy", "multiply by beta instead of dividing", [("+ r / beta", "+ r * beta")]),
    ("1-closed-form-policy", "unstable partition function", [("log_Z = float(m + np.log(np.sum(np.exp(log_weights - m))))", "log_Z = float(np.log(np.sum(np.exp(log_weights))))")]),
    ("1-closed-form-policy", "implicit reward without beta", [("return beta * (np.log(", "return 1.0 * (np.log(")]),
    ("1-closed-form-policy", "implicit reward flips the ratio", [("np.log(np.asarray(pi, dtype=float)) - np.log(np.asarray(ref_probs, dtype=float))", "np.log(np.asarray(ref_probs, dtype=float)) - np.log(np.asarray(pi, dtype=float))")]),
    ("2-dpo-loss", "reference terms dropped", [("(pi_w - ref_w) - (pi_l - ref_l)", "pi_w - pi_l")]),
    ("2-dpo-loss", "reference added instead of subtracted", [("(pi_w - ref_w) - (pi_l - ref_l)", "(pi_w + ref_w) - (pi_l + ref_l)")]),
    ("2-dpo-loss", "winner and loser swapped", [("(pi_w - ref_w) - (pi_l - ref_l)", "(pi_l - ref_l) - (pi_w - ref_w)")]),
    ("2-dpo-loss", "beta forgotten", [("return beta * ((pi_w - ref_w)", "return 1.0 * ((pi_w - ref_w)")]),
    ("2-dpo-loss", "log(1 + exp(-m)) overflows", [("np.logaddexp(0.0, -margin)", "np.log(1.0 + np.exp(-margin))")]),
    ("2-dpo-loss", "returns the batch mean", [("return np.logaddexp(0.0, -margin)", "return np.mean(np.logaddexp(0.0, -margin))")]),
    ("3-tabular-gradient", "weight uses the winner's reward first", [("sigmoid(rhat[l] - rhat[w])", "sigmoid(rhat[w] - rhat[l])")]),
    ("3-tabular-gradient", "no weight (naive likelihood ratio)", [("-beta * weight * (grad_log_w", "-beta * (grad_log_w")]),
    ("3-tabular-gradient", "beta missing from the gradient", [("-beta * weight * (grad_log_w", "-weight * (grad_log_w")]),
    ("3-tabular-gradient", "sign flipped (ascent direction)", [("-beta * weight * (grad_log_w", "beta * weight * (grad_log_w")]),
    ("3-tabular-gradient", "sum over pairs instead of mean", [("return grad / len(pairs)", "return grad")]),
    ("3-tabular-gradient", "only the winner's log-prob gradient", [("(grad_log_w - grad_log_l)", "grad_log_w")]),
    ("4-train-toy-policy", "in-place update corrupts the reference", [("theta = np.array(ref_logits, dtype=float).copy()", "theta = ref_logits"), ("theta = theta - lr * mean_grad(theta, ref_logits, C, beta)", "theta -= lr * mean_grad(theta, ref_logits, C, beta)")]),
    ("4-train-toy-policy", "gradient ascent", [("theta = theta - lr * mean_grad(theta, ref_logits, C, beta)", "theta = theta + lr * mean_grad(theta, ref_logits, C, beta)")]),
    ("4-train-toy-policy", "learning rate ignored", [("theta = theta - lr * mean_grad(theta, ref_logits, C, beta)", "theta = theta - mean_grad(theta, ref_logits, C, beta)")]),
    ("4-train-toy-policy", "returns the reference instead of theta", [("return theta, losses", "return np.array(ref_logits, dtype=float), losses")]),
    ("4-train-toy-policy", "starts from zeros instead of the reference", [("theta = np.array(ref_logits, dtype=float).copy()", "theta = np.zeros(len(ref_logits))")]),
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
    for entry in MUTANTS:
        folder, name, edits = entry
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
