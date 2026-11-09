#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/lstm/scripts/check_mutants.py 
"""

import importlib.util
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
    ("1-error-backflow", "derivative taken at h_{t-1} instead of h_t", [("step = np.diag(1.0 - h**2) @ W", "step = np.diag(1.0 - h_prev**2) @ W"), ("        h = np.tanh(W @ h + U @ x)\n", "        h_prev = h\n        h = np.tanh(W @ h + U @ x)\n")]),
    ("1-error-backflow", "tanh derivative dropped (linear RNN Jacobian)", [("np.diag(1.0 - h**2) @ W", "W")]),
    ("1-error-backflow", "chain product in the wrong order", [("J = step @ J", "J = J @ step")]),
    ("1-error-backflow", "W transposed", [("np.diag(1.0 - h**2) @ W", "np.diag(1.0 - h**2) @ W.T")]),
    ("1-error-backflow", "CEC with the sigmoid factor 0.25", [("step = np.eye(n) * 1.0 * 1.0", "step = np.eye(n) * 0.25")]),
    ("1-error-backflow", "lag counted from 0", [("enumerate(norms, start=1)", "enumerate(norms, start=0)")]),
    ("1-error-backflow", "lag_below uses <= instead of <", [("if value < threshold", "if value <= threshold")]),
    ("2-memory-cell", "g with range (-1, 1) instead of (-2, 2)", [("return 4.0 / (1.0 + np.exp(-x)) - 2.0", "return 2.0 / (1.0 + np.exp(-x)) - 1.0")]),
    ("2-memory-cell", "g is the plain logistic sigmoid", [("return 4.0 / (1.0 + np.exp(-x)) - 2.0", "return 1.0 / (1.0 + np.exp(-x))")]),
    ("2-memory-cell", "state forgets (later-LSTM style decay of 0.9)", [("s = s_prev + i_gate", "s = 0.9 * s_prev + i_gate")]),
    ("2-memory-cell", "state overwritten instead of accumulated", [("s = s_prev + i_gate", "s = 0 * s_prev + i_gate")]),
    ("2-memory-cell", "output gate omitted", [("y = o_gate * h(s)", "y = h(s)")]),
    ("2-memory-cell", "output computed from the old state", [("y = o_gate * h(s)", "y = o_gate * h(s_prev)")]),
    ("2-memory-cell", "input gate omitted", [("s = s_prev + i_gate * g(net_c)", "s = s_prev + g(net_c)")]),
    ("2-memory-cell", "previous output not fed back", [("z = np.concatenate([x, y])", "z = np.concatenate([x, np.zeros(n)])")]),
    ("2-memory-cell", "state reset at every step", [("        y, s = memory_cell_step(z, s, params)", "        y, s = memory_cell_step(z, np.zeros(n), params)")]),
    ("3-adding-problem", "input gate opens for every nonzero marker", [("1.0 if marker == 1.0 else 0.0", "1.0 if marker != 0.0 else 0.0")]),
    ("3-adding-problem", "state overwritten (keeps only the last marked value)", [("s = s + input_gate * value", "s = input_gate * value if input_gate else s")]),
    ("3-adding-problem", "scale 1/2 instead of 1/4", [("return 0.5 + s / 4.0", "return 0.5 + s / 2.0")]),
    ("3-adding-problem", "length upper bound excluded", [("T + T // 10 + 1", "T + T // 10")]),
    ("3-adding-problem", "second marked pair may equal the first", [("candidates = [j for j in range(T // 2 - 1) if j != first]", "candidates = list(range(T // 2 - 1))")]),
    ("3-adding-problem", "marked first pair keeps its value", [("        if j == 0:\n            X[j, 0] = 0.0\n", "")]),
    ("3-adding-problem", "no -1 marker on the last pair", [("    X[-1, 1] = -1.0\n", "")]),
    ("3-adding-problem", "target scale 1/2", [("(X[first, 0] + X[second, 0]) / 4.0", "(X[first, 0] + X[second, 0]) / 2.0")]),
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
