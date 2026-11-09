#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/resnet/scripts/check_mutants.py
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
    ("1-residual-block", "ReLU before the addition, not after", [("return relu(F + shortcut)", "return relu(F) + shortcut")]),
    ("1-residual-block", "no ReLU after the addition", [("return relu(F + shortcut)", "return F + shortcut")]),
    ("1-residual-block", "no ReLU inside F", [("F = W2 @ relu(W1 @ x)", "F = W2 @ (W1 @ x)")]),
    ("1-residual-block", "projection ignored (always the identity)", [("shortcut = x if Ws is None else Ws @ x", "shortcut = x")]),
    ("2-block-backward", "shortcut term dropped from g_x", [("g_x = W1.T @ g_a + g_z", "g_x = W1.T @ g_a")]),
    ("2-block-backward", "output ReLU does not gate the gradient", [("g_z = g_out * (z > 0)", "g_z = g_out")]),
    ("2-block-backward", "inner ReLU does not gate the gradient", [("g_a = g_r * (a > 0)", "g_a = g_r")]),
    ("2-block-backward", "shortcut gradient counted twice", [("g_x = W1.T @ g_a + g_z", "g_x = W1.T @ g_a + 2 * g_z")]),
    ("3-gradient-norms", "backward pass starts from ones, not h_n", [("g = h.copy()", "g = np.ones_like(h)")]),
    ("3-gradient-norms", "norms left in output-to-input order", [("return np.array(norms[::-1])", "return np.array(norms)")]),
    ("3-gradient-norms", "input norm missing (n entries)", [("return np.array(norms[::-1])", "return np.array(norms[::-1][1:])")]),
    ("3-gradient-norms", "caches paired with the wrong layers", [("zip(reversed(net), reversed(caches))", "zip(reversed(net), caches)")]),
    ("4-bottleneck-params", "kernel size not squared", [("return k * k * c_in * c_out", "return k * c_in * c_out")]),
    ("4-bottleneck-params", "basic block counts one convolution", [("return 2 * conv_params(3, c, c)", "return conv_params(3, c, c)")]),
    ("4-bottleneck-params", "bottleneck middle layer is 1x1", [("conv_params(3, c_mid, c_mid)", "conv_params(1, c_mid, c_mid)")]),
    ("4-bottleneck-params", "projection joins the narrow end", [("return conv_params(1, c_out, c_out)", "return conv_params(1, c_out, c_out // 4)")]),
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
        folder, name, edits = entry[:3]
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
