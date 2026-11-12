#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/td3/scripts/check_mutants.py
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
    ("1-clipped-target", "mean instead of minimum", [("np.minimum(q1_next, q2_next)", "(q1_next + q2_next) / 2")]),
    ("1-clipped-target", "maximum instead of minimum", [("np.minimum(q1_next, q2_next)", "np.maximum(q1_next, q2_next)")]),
    ("1-clipped-target", "only the first critic", [("np.minimum(q1_next, q2_next)", "q1_next")]),
    ("1-clipped-target", "terminal states still bootstrap", [("(1.0 - done) * ", "")]),
    ("1-clipped-target", "terminal mask inverted", [("(1.0 - done) * ", "done * ")]),
    ("1-clipped-target", "gamma also scales the reward", [("r + gamma * (1.0 - done) * q_min", "gamma * (r + (1.0 - done) * q_min)")]),
    ("1-clipped-target", "loss trains one critic only", [("np.mean((y - q1) ** 2) + np.mean((y - q2) ** 2)", "np.mean((y - q1) ** 2)")]),
    ("1-clipped-target", "loss averages the two critics", [("np.mean((y - q1) ** 2) + np.mean((y - q2) ** 2)", "(np.mean((y - q1) ** 2) + np.mean((y - q2) ** 2)) / 2")]),
    ("2-target-smoothing", "noise is not clipped", [("np.clip(sigma * z, -c, c)", "sigma * z")]),
    ("2-target-smoothing", "z is clipped before scaling", [("np.clip(sigma * z, -c, c)", "sigma * np.clip(z, -c, c)")]),
    ("2-target-smoothing", "action bounds are not enforced", [("np.clip(mu_next + noise, low, high)", "mu_next + noise")]),
    ("2-target-smoothing", "only the bounds are applied", [("np.clip(sigma * z, -c, c)", "sigma * z"), ("np.clip(mu_next + noise, low, high)", "np.clip(mu_next + noise, low, high)")]),
    ("2-target-smoothing", "noise ignores sigma", [("np.clip(sigma * z, -c, c)", "np.clip(z, -c, c)")]),
    ("2-target-smoothing", "sampler draws uniform noise", [("rng.standard_normal(mu_next.shape)", "rng.uniform(-1, 1, mu_next.shape)")]),
    ("3-overestimation-demo", "double estimate reuses the same critic", [("b_at = qb[rows, star]", "b_at = qa[rows, star]")]),
    ("3-overestimation-demo", "clipped takes the maximum", [("np.minimum(a_at, b_at)", "np.maximum(a_at, b_at)")]),
    ("3-overestimation-demo", "max is taken on the wrong critic's argmax", [("star = qa.argmax(axis=1)", "star = np.zeros(trials, dtype=int)")]),
    ("3-overestimation-demo", "noise ignores sigma", [("rng.normal(0.0, sigma, (trials, n_actions))", "rng.normal(0.0, 1.0, (trials, n_actions))")]),
    ("3-overestimation-demo", "ignores the seed", [("np.random.default_rng(seed)", "np.random.default_rng()")]),
    ("3-overestimation-demo", "max is a mean over actions", [('float(qa.max(axis=1).mean())', 'float(qa.mean())')]),
    ("4-delayed-updates", "steps are 0-based", [("range(1, total_steps + 1)", "range(0, total_steps)")]),
    ("4-delayed-updates", "updates on every step", [("if t % d == 0", "if True")]),
    ("4-delayed-updates", "off by one multiple", [("t % d == 0", "t % d == 1")]),
    ("4-delayed-updates", "gap ignores the delay", [("k = len(actor_update_steps(total_steps, d))", "k = total_steps")]),
    ("4-delayed-updates", "gap ignores the start gap", [("return gap0 * (1.0 - tau) ** k", "return (1.0 - tau) ** k")]),
    ("4-delayed-updates", "tau instead of 1 - tau", [("(1.0 - tau) ** k", "tau ** k")]),
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
