#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/christiano/scripts/check_mutants.py
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
    ("1-preference-loss", "naive exp overflows on large gaps", [("np.exp(_log_sigmoid(diff))", "1.0 / (1.0 + np.exp(-diff))"), ("return -np.logaddexp(0.0, -x)", "return np.log(1.0 / (1.0 + np.exp(-x)))")]),
    ("1-preference-loss", "mu oriented toward segment 2", [("-(mu * lp1 + (1.0 - mu) * lp2)", "-((1.0 - mu) * lp1 + mu * lp2)")]),
    ("1-preference-loss", "eps mixes in eps, not eps/2", [("(1.0 - eps) * p + eps / 2.0", "(1.0 - eps) * p + eps")]),
    ("1-preference-loss", "sum instead of mean over the batch", [("return float(np.mean(-(mu", "return float(np.sum(-(mu")]),
    ("1-preference-loss", "tie labels rounded to a hard label", [("mu = np.asarray(mu, dtype=float)\n    if", "mu = (np.asarray(mu, dtype=float) > 0.5).astype(float)\n    if")]),
    ("1-preference-loss", "eps ignored in the loss", [("lp1 = np.log(preference_prob(ret1, ret2, eps))", "lp1 = _log_sigmoid(ret1 - ret2)"), ("lp2 = np.log(preference_prob(ret2, ret1, eps))", "lp2 = _log_sigmoid(ret2 - ret1)")]),
    ("2-fit-linear-reward", "mean over time instead of sum", [("F1.sum(axis=1)", "F1.mean(axis=1)"), ("F2.sum(axis=1)", "F2.mean(axis=1)")]),
    ("2-fit-linear-reward", "gradient sign flipped", [("(sigmoid(diff) - mu)", "(mu - sigmoid(diff))")]),
    ("2-fit-linear-reward", "label mass on segment 2", [("(sigmoid(diff) - mu)", "(sigmoid(diff) - (1.0 - mu))")]),
    ("2-fit-linear-reward", "only segment 1 enters the gradient", [("(S1 - S2)).mean", "S1).mean")]),
    ("2-fit-linear-reward", "gradient ascent instead of descent", [("w = w - lr * loss_grad", "w = w + lr * loss_grad")]),
    ("2-fit-linear-reward", "fit stops after one step", [("for _ in range(steps):", "for _ in range(1):")]),
    ("3-ensemble-queries", "average first, normalize second", [("return np.mean([normalize_rewards(F @ w) for w in ws], axis=0)", "return normalize_rewards(np.mean([F @ w for w in ws], axis=0))")]),
    ("3-ensemble-queries", "normalize without centering", [("return (r - r.mean()) / sd", "return r / sd")]),
    ("3-ensemble-queries", "constant input divides by zero", [("    if sd == 0:\n        return np.zeros_like(r)\n", "")]),
    ("3-ensemble-queries", "variance over pairs, not members", [("probs.var(axis=0)", "np.repeat(probs.var(axis=1).mean(), probs.shape[1])")]),
    ("3-ensemble-queries", "lowest variance first", [("np.argsort(-var, kind=\"stable\")", "np.argsort(var, kind=\"stable\")")]),
    ("3-ensemble-queries", "variance of return differences", [("probs = np.array([sigmoid(segment_returns(F1, w) - segment_returns(F2, w)) for w in ws])", "probs = np.array([segment_returns(F1, w) - segment_returns(F2, w) for w in ws])")]),
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
