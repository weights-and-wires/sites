#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/rectified-flow/scripts/check_mutants.py
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
K1 = "1-kernel-velocity"
K2 = "2-straightness"
K3 = "3-reflow"
KERNEL_END = "return (w @ x1 - z) / (1.0 - t)"
D2 = "d2 = ((z[:, None, :] - xt[None, :, :]) ** 2).sum(axis=-1)"
XT = "xt = t * x1 + (1.0 - t) * x0"
EULER = "z = z + step * kernel_velocity(z, k * step, x0, x1, h)"

MUTANTS = [
    (K1, "drops the 1 / (1 - t) factor", [(KERNEL_END, "return (w @ x1 - z)")]),
    (K1, "averages the line directions x1 - x0 and ignores z", [(KERNEL_END, "return w @ (x1 - x0)")]),
    (K1, "kernel centered at x1 instead of x_t", [(XT, "xt = x1")]),
    (K1, "kernel centered at x0 instead of x_t", [(XT, "xt = x0")]),
    (K1, "bandwidth used as a variance (h instead of h^2)", [("-0.5 * d2 / h**2", "-0.5 * d2 / h")]),
    (K1, "weights not normalized", [("w = w / w.sum(axis=1, keepdims=True)", "pass")]),
    (K1, "no max subtraction in the softmax", [("np.exp(logw - logw.max(axis=1, keepdims=True))", "np.exp(logw)")]),
    (K1, "distance averaged over coordinates instead of summed", [(D2, D2.replace(".sum(axis=-1)", ".mean(axis=-1)"))]),
    (K2, "norm not squared", [("((chord[None] - rate) ** 2).sum(axis=-1)", "np.sqrt(((chord[None] - rate) ** 2).sum(axis=-1))")]),
    (K2, "forgets the factor h (sum grows with the step count)", [("return float(h * per_step.sum())", "return float(per_step.sum())")]),
    (K2, "chord taken from the last step instead of the endpoints", [("chord = path[-1] - path[0]", "chord = path[-1] - path[-2]")]),
    (K2, "coordinates averaged instead of summed", [(".sum(axis=-1).mean(axis=1)", ".mean(axis=-1).mean(axis=1)")]),
    (K2, "trajectories summed instead of averaged", [(".mean(axis=1)", ".sum(axis=1)")]),
    (K2, "velocity is the raw increment, not divided by h", [("rate = (path[1:] - path[:-1]) / h", "rate = (path[1:] - path[:-1])")]),
    (K3, "Euler evaluates the field at the end of the step", [(EULER, EULER.replace("k * step", "(k + 1) * step"))]),
    (K3, "reflow returns the old pairs", [("return np.array(x0, dtype=float), euler_path(x0, x1, h, n_steps)[-1]", "return np.array(x0, dtype=float), np.array(x1, dtype=float)")]),
    (K3, "reflow scrambles the pairing of the endpoints", [("euler_path(x0, x1, h, n_steps)[-1]", "euler_path(x0, x1, h, n_steps)[-1][::-1]")]),
    (K3, "trajectories start at the data x1 instead of the noise x0", [("z = np.array(x0, dtype=float)\n    path", "z = np.array(x1, dtype=float)\n    path")]),
    (K3, "step size 1 / (n + 1)", [("step = 1.0 / n_steps", "step = 1.0 / (n_steps + 1)")]),
    (K3, "steps against the field (backward sampling without flipping time)", [(EULER, EULER.replace("z + step", "z - step"))]),
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
