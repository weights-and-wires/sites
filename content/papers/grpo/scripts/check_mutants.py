#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/grpo/scripts/check_mutants.py
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
    ("1-group-advantages", "no division by the std (mean baseline only)", [("return (r - mean) / (std + eps)", "return r - mean")]),
    ("1-group-advantages", "sample std (ddof=1) instead of population std", [("r.std(axis=-1, keepdims=True)", "r.std(axis=-1, keepdims=True, ddof=1)")]),
    ("1-group-advantages", "no eps, so a constant group gives nan", [("(std + eps)", "std")]),
    ("1-group-advantages", "statistics over the batch axis, not the group axis", [
        ("mean = r.mean(axis=-1, keepdims=True)", "mean = r.mean(axis=0, keepdims=True)"),
        ("std = r.std(axis=-1, keepdims=True)", "std = r.std(axis=0, keepdims=True)")]),
    ("1-group-advantages", "one mean and std for the whole batch", [
        ("mean = r.mean(axis=-1, keepdims=True)", "mean = r.mean()"),
        ("std = r.std(axis=-1, keepdims=True)", "std = r.std()")]),
    ("2-grpo-objective", "KL ratio inverted: pi/pi_ref instead of pi_ref/pi", [("d = logp_ref - logp", "d = logp - logp_ref")]),
    ("2-grpo-objective", "KL estimator without the -1", [("return np.exp(d) - d - 1.0", "return np.exp(d) - d")]),
    ("2-grpo-objective", "global token mean instead of per-output 1/|o_i|", [
        ("per_output = (per_token * mask).sum(axis=1) / mask.sum(axis=1)\n    return per_output.mean()",
         "return (per_token * mask).sum() / mask.sum()")]),
    ("2-grpo-objective", "padding tokens are not masked", [("(per_token * mask).sum(axis=1) / mask.sum(axis=1)", "per_token.mean(axis=1)")]),
    ("2-grpo-objective", "KL added to the objective instead of subtracted", [("clipped - beta * kl_k3", "clipped + beta * kl_k3")]),
    ("2-grpo-objective", "min dropped: only the clipped term", [("np.minimum(ratio * a, np.clip(ratio, 1 - eps, 1 + eps) * a)", "np.clip(ratio, 1 - eps, 1 + eps) * a")]),
    ("2-grpo-objective", "max instead of min", [("np.minimum(ratio * a,", "np.maximum(ratio * a,")]),
    ("3-process-advantages", "forward cumulative sum instead of the sum over later steps", [("np.cumsum(norm[::-1])[::-1]", "np.cumsum(norm)")]),
    ("3-process-advantages", "each output normalized on its own", [
        ("norm = (np.asarray(s, dtype=float) - mean) / (std + 1e-6)",
         "s = np.asarray(s, dtype=float)\n        norm = (s - s.mean()) / (s.std() + 1e-6)")]),
    ("3-process-advantages", "no cumulative sum: each step gets only its own reward", [("np.cumsum(norm[::-1])[::-1]", "norm")]),
    ("3-process-advantages", "no standard deviation in the normalization", [("/ (std + 1e-6)", "")]),
    ("4-bandit-update", "KL coefficient inverted: pi/pi_ref - 1", [("pi_ref[actions] / pi[actions] - 1.0", "pi[actions] / pi_ref[actions] - 1.0")]),
    ("4-bandit-update", "KL term has the wrong sign", [("adv + beta * (pi_ref", "adv - beta * (pi_ref")]),
    ("4-bandit-update", "gradient of log pi without the - pi term", [("(onehot - pi)", "onehot")]),
    ("4-bandit-update", "gradient descent instead of ascent", [("return logits + lr * grad", "return logits - lr * grad")]),
    ("4-bandit-update", "advantage not used: raw rewards as the coefficient", [("adv = group_advantages(rewards)", "adv = np.asarray(rewards, dtype=float)")]),
    ("4-bandit-update", "sum over the group instead of mean", [(".mean(axis=0)", ".sum(axis=0)")]),
    ("4-bandit-update", "updates the input logits in place", [("return logits + lr * grad", "logits += lr * grad\n    return logits")]),
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
