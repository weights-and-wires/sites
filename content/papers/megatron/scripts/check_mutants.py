#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/megatron/scripts/check_mutants.py
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
    ("1-column-row-mlp", "GeLU applied after the reduction", [("partials = [gelu(X @ Ai) @ Bi", "partials = [(X @ Ai) @ Bi"), ("return reduce_fn(partials)", "return gelu(reduce_fn(partials))")]),
    ("1-column-row-mlp", "GeLU forgotten", [("gelu(X @ Ai) @ Bi", "(X @ Ai) @ Bi")]),
    ("1-column-row-mlp", "two all-reduces instead of one", [("    return reduce_fn(partials)", "    reduce_fn(partials)\n    return reduce_fn(partials)")]),
    ("1-column-row-mlp", "A split by rows instead of columns", [("return np.split(A, n, axis=1)", "return np.split(A, n, axis=0)")]),
    ("2-backward-f-operator", "no all-reduce: GPU 0's local dX", [("return all_reduce(dX_local), ", "return dX_local[0], ")]),
    ("2-backward-f-operator", "all-reduce averages instead of sums", [("return all_reduce(dX_local), ", "return all_reduce(dX_local) / n, ")]),
    ("2-backward-f-operator", "dB uses the pre-activation H", [("dB_blocks.append(Y.T @ dZ)", "dB_blocks.append(H.T @ dZ)")]),
    ("2-backward-f-operator", "GeLU derivative forgotten", [("dH = dY * dgelu(H)", "dH = dY")]),
    ("3-parallel-attention", "GPU treats its heads as one wide head", [("heads // n, d_k, mask)", "1, cols, mask)")]),
    ("3-parallel-attention", "W^O rows always from the first block", [("@ Wo[s]", "@ Wo[:cols]")]),
    ("3-parallel-attention", "keys projected with the query slice", [("Wk[:, s]", "Wq[:, s]")]),
    ("3-parallel-attention", "strided columns split heads apart", [("s = slice(r * cols, (r + 1) * cols)", "s = np.arange(r, d, n)")]),
    ("3-parallel-attention", "mask not forwarded", [("heads // n, d_k, mask)", "heads // n, d_k)")]),
    ("3-parallel-attention", "no divisibility check", [("if heads % n != 0:", "if False:")]),
    ("4-communication-volume", "2 all-reduces per layer instead of 4", [("4 * b * s * h", "2 * b * s * h")]),
    ("4-communication-volume", "ring factor without the 2", [("2 * (t - 1) * elements / t", "(t - 1) * elements / t")]),
    ("4-communication-volume", "ring factor 2 regardless of t", [("2 * (t - 1) * elements / t", "2 * elements")]),
    ("4-communication-volume", "fusion flag ignored", [("b * s if fused else b * s * v", "b * s * v")]),
    ("4-communication-volume", "vocabulary rounded down", [("-(-v // m) * m", "(v // m) * m")]),
    ("4-communication-volume", "vocabulary always grows by one block", [("-(-v // m) * m", "(v // m + 1) * m")]),
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
