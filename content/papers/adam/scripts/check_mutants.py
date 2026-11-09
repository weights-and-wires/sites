#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/adam/scripts/check_mutants.py
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
    ("1-adam-step", "no bias correction on m", [("m_hat = m_new / (1 - beta1**t)", "m_hat = m_new")]),
    ("1-adam-step", "no bias correction on v", [("v_hat = v_new / (1 - beta2**t)", "v_hat = v_new")]),
    ("1-adam-step", "epsilon inside the square root", [("np.sqrt(v_hat) + eps", "np.sqrt(v_hat + eps)")]),
    ("1-adam-step", "no square root (divide by v_hat)", [("np.sqrt(v_hat) + eps", "v_hat + eps")]),
    ("1-adam-step", "v tracks g instead of g squared", [("(1 - beta2) * g * g", "(1 - beta2) * g")]),
    ("1-adam-step", "loop timestep starts at 0", [("range(1, steps + 1)", "range(0, steps)")]),
    ("1-adam-step", "in-place update of m and v", [("m_new = beta1 * m + (1 - beta1) * g", "m *= beta1; m += (1 - beta1) * g; m_new = m")]),
    ("2-bias-correction", "corrects only the first moment", [("v_hat = v / (1 - beta2**t)", "v_hat = v")]),
    ("2-bias-correction", "corrects only the second moment", [("m_hat = m / (1 - beta1**t)", "m_hat = m")]),
    ("2-bias-correction", "betas swapped in the correction", [("m / (1 - beta1**t)", "m / (1 - beta2**t)"), ("v / (1 - beta2**t)", "v / (1 - beta1**t)")]),
    ("2-bias-correction", "exponent t - 1 (off by one)", [("1 - beta1**t", "1 - beta1 ** (t - 1)")]),
    ("2-bias-correction", "bias_correction flag ignored", [("if bias_correction:", "if True:")]),
    ("3-rosenbrock", "gradient: wrong cross term (200 -> 100)", [("200 * (y - x**2)", "100 * (y - x**2)")]),
    ("3-rosenbrock", "gradient: missing factor x in the first term", [("400 * x * (y - x**2)", "400 * (y - x**2)")]),
    ("3-rosenbrock", "momentum uses the EMA form (1 - mu) * g", [("v = mu * v + grad_fn(x)", "v = mu * v + (1 - mu) * grad_fn(x)")]),
    ("3-rosenbrock", "momentum steps along the raw gradient", [("x = x - lr * v", "x = x - lr * grad_fn(x)")]),
    ("3-rosenbrock", "momentum ascends", [("x = x - lr * v", "x = x + lr * v")]),
    ("3-rosenbrock", "momentum overwrites x0 in place", [("x = np.array(x0, dtype=float)\n    v = np.zeros_like(x)\n    for _ in range(steps):", "x = x0\n    v = np.zeros_like(x)\n    for _ in range(steps):"), ("x = x - lr * v\n    return x", "x -= lr * v\n    return x")]),
    ("4-adamax", "u uses g without the absolute value", [("np.maximum(beta2 * u, np.abs(g))", "np.maximum(beta2 * u, g)")]),
    ("4-adamax", "u is an average instead of a max", [("np.maximum(beta2 * u, np.abs(g))", "beta2 * u + (1 - beta2) * np.abs(g)")]),
    ("4-adamax", "no bias correction on m", [("(alpha / (1 - beta1**t)) * m_new", "alpha * m_new")]),
    ("4-adamax", "u not decayed (running max)", [("np.maximum(beta2 * u, np.abs(g))", "np.maximum(u, np.abs(g))")]),
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
    skip_torch = False
    survivors = 0
    ran = 0
    for entry in MUTANTS:
        folder, name, edits = entry[:3]
        if len(entry) == 4 and skip_torch:
            print(f"SKIP  {folder}: {name} (torch not available)")
            continue
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
