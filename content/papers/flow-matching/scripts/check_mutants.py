#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/flow-matching/scripts/check_mutants.py
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
    ("1-ot-conditional-target", "target drops sigma_min (uses x1 - x0)",
     [("ut = x1 - (1.0 - sigma_min) * x0", "ut = x1 - x0")]),
    ("1-ot-conditional-target", "standard deviation 1 - t instead of 1 - (1 - sigma_min) t",
     [("xt = (1.0 - (1.0 - sigma_min) * t) * x0 + t * x1", "xt = (1.0 - t) * x0 + t * x1")]),
    ("1-ot-conditional-target", "target uses the OT path's data term with the wrong sign on x0",
     [("ut = x1 - (1.0 - sigma_min) * x0", "ut = x1 + (1.0 - sigma_min) * x0")]),
    ("1-ot-conditional-target", "loss averages over the d coordinates instead of summing",
     [("np.sum(diff**2, axis=-1)", "np.mean(diff**2, axis=-1)")]),
    ("1-ot-conditional-target", "loss evaluates v at x0 instead of x_t",
     [("diff = v(xt, t) - ut", "diff = v(x0, t) - ut")]),
    ("2-gaussian-path-vf", "Theorem 3 drops the sigma'/sigma factor",
     [("return dsigma(t) / sigma(t) * (x - mu(t, x1)) + dmu(t, x1)", "return (x - mu(t, x1)) + dmu(t, x1)")]),
    ("2-gaussian-path-vf", "Theorem 3 forgets to subtract the mean",
     [("(x - mu(t, x1))", "x")]),
    ("2-gaussian-path-vf", "VP mean derivative has the wrong sign (uses d/ds, not d/dt)",
     [("return 0.5 * beta(1.0 - t) * alpha(1.0 - t) * x1", "return -0.5 * beta(1.0 - t) * alpha(1.0 - t) * x1")]),
    ("2-gaussian-path-vf", "VP sigma derivative has the wrong sign",
     [("return -a * (0.5 * beta(1.0 - t) * a) / sigma(t)", "return a * (0.5 * beta(1.0 - t) * a) / sigma(t)")]),
    ("2-gaussian-path-vf", "VP derivative evaluates beta at t instead of 1 - t",
     [("return 0.5 * beta(1.0 - t) * alpha(1.0 - t) * x1", "return 0.5 * beta(t) * alpha(1.0 - t) * x1")]),
    ("3-marginal-field", "Gaussians centered at x1 instead of t * x1",
     [("(x[:, None, :] - t * data[None]) ** 2", "(x[:, None, :] - data[None]) ** 2")]),
    ("3-marginal-field", "weights use sigma instead of sigma squared",
     [("logw = -0.5 * d2 / sigma**2", "logw = -0.5 * d2 / sigma")]),
    ("3-marginal-field", "weights not normalized",
     [("w = w / w.sum(axis=1, keepdims=True)", "pass")]),
    ("3-marginal-field", "no max subtraction in the softmax",
     [("np.exp(logw - logw.max(axis=1, keepdims=True))", "np.exp(logw)")]),
    ("3-marginal-field", "uniform average of conditional fields (ignores x in the weights)",
     [("w = np.exp(logw - logw.max(axis=1, keepdims=True))", "w = np.ones_like(logw)")]),
    ("3-marginal-field", "conditional field drops the 1 / sigma_t factor",
     [("u = (data[None] - (1.0 - sigma_min) * x[:, None, :]) / sigma", "u = (data[None] - (1.0 - sigma_min) * x[:, None, :])")]),
    ("4-ode-solvers", "Euler evaluates the field at the end of the step",
     [("x = x + h * v(x, k * h)", "x = x + h * v(x, (k + 1) * h)")]),
    ("4-ode-solvers", "midpoint evaluates v at the start time instead of t + h/2",
     [("x = x + h * v(x_half, t + 0.5 * h)", "x = x + h * v(x_half, t)")]),
    ("4-ode-solvers", "midpoint half step uses the full step (x + h v)",
     [("x_half = x + 0.5 * h * v(x, t)", "x_half = x + h * v(x, t)")]),
    ("4-ode-solvers", "midpoint ignores the half step (is Euler with a shifted time)",
     [("x_half = x + 0.5 * h * v(x, t)", "x_half = x")]),
    ("4-ode-solvers", "Euler integrates to t = 1 - 1/n (one step short)",
     [("for k in range(n_steps):\n        x = x + h * v(x, k * h)", "for k in range(n_steps - 1):\n        x = x + h * v(x, k * h)")]),
    ("4-ode-solvers", "Euler mutates the caller's array",
     [("x = np.array(x0, dtype=float)\n    h = 1.0 / n_steps\n    for k in range(n_steps):\n        x = x + h * v(x, k * h)",
       "x = x0\n    h = 1.0 / n_steps\n    for k in range(n_steps):\n        x += h * v(x, k * h)")]),
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
