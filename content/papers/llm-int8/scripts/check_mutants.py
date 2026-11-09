#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/llm-int8/scripts/check_mutants.py
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
    ("1-vectorwise-matmul", "floor instead of round-to-nearest", [("np.rint(A * c[:, None])", "np.floor(A * c[:, None])")]),
    ("1-vectorwise-matmul", "int8 accumulation (overflow)", [("Xq.astype(np.int32) @ Wq.astype(np.int32)", "Xq @ Wq")]),
    ("1-vectorwise-matmul", "one tensor-wide constant for X", [
        ("c = 127.0 / np.max(np.abs(A), axis=1)", "c = np.full(A.shape[0], 127.0 / np.max(np.abs(A)))")]),
    ("1-vectorwise-matmul", "W scaled per row instead of per column", [
        ("Wt, cw = quantize_rows(W.T)\n    Wq = Wt.T", "Wq, cw = quantize_rows(W)")]),
    ("1-vectorwise-matmul", "skip the c_w denormalization", [("np.outer(cx, cw)", "cx[:, None]")]),
    ("1-vectorwise-matmul", "scale to 128 instead of 127", [("c = 127.0 /", "c = 128.0 /")]),
    ("2-find-outliers", "reduces over rows instead of columns", [("axis=0", "axis=1")]),
    ("2-find-outliers", "misses negative outliers (no abs)", [("np.max(np.abs(X), axis=0)", "np.max(X, axis=0)")]),
    ("2-find-outliers", "mean instead of max", [("np.max(np.abs(X), axis=0)", "np.mean(np.abs(X), axis=0)")]),
    ("2-find-outliers", "strict inequality at the threshold", [(">= alpha", "> alpha")]),
    ("2-find-outliers", "hard-coded threshold", [(">= alpha", ">= 6.0")]),
    ("3-mixed-decomposition", "outlier part uses all rows of W", [("X[:, O] @ W[O, :]", "X[:, O] @ W[: len(O), :]")]),
    ("3-mixed-decomposition", "outlier columns also stay in the int8 part", [(
        "vectorwise_matmul(X[:, regular], W[regular, :])", "vectorwise_matmul(X, W)")]),
    ("3-mixed-decomposition", "outlier part is dropped", [("out = X[:, O] @ W[O, :]", "out = np.zeros((X.shape[0], W.shape[1]))")]),
    ("3-mixed-decomposition", "outlier columns paired with regular rows of W", [("out = X[:, O] @ W[O, :]", "out = X[:, O] @ W[regular[: len(O)], :]")]),
    ("3-mixed-decomposition", "no guard when every dimension is an outlier", [("if regular.size:", "if True:")]),
    ("3-mixed-decomposition", "outliers quantized to int8 as well", [("out = X[:, O] @ W[O, :]", "out = vectorwise_matmul(X[:, O], W[O, :]) if O.size else 0.0")]),
    ("3-mixed-decomposition", "writes into X", [("O = outlier_dims(X, alpha)", "O = outlier_dims(X, alpha)\n    X[:, O] = 0.0")]),
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
