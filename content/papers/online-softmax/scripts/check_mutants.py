#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/online-softmax/scripts/check_mutants.py
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
ONLINE = "d = d * math.exp(m - m_new) + math.exp(x - m_new)"
MUTANTS = [
    ("1-online-normalizer", "forget to rescale d when the max changes", [(ONLINE, "d = d + math.exp(x - m_new)")]),
    ("1-online-normalizer", "rescale in the wrong direction", [(ONLINE, "d = d * math.exp(m_new - m) + math.exp(x - m_new)")]),
    ("1-online-normalizer", "new term uses the old max", [(ONLINE, "d = d * math.exp(m - m_new) + math.exp(x - m)")]),
    ("1-online-normalizer", "running max starts at 0", [("m, d = -math.inf, 0.0\n    for x in xs", "m, d = 0.0, 0.0\n    for x in xs")]),
    ("1-online-normalizer", "max found in a first pass (two passes)", [("m, d = -math.inf, 0.0\n    for x in xs", "m, d = max(xs), 0.0\n    for x in xs")]),
    ("1-online-normalizer", "naive sum, no max subtraction", [(ONLINE, "d = d + math.exp(x)")]),
    ("2-merge-blocks", "left summary not rescaled", [("d_a * math.exp(m_a - m) + d_b", "d_a + d_b")]),
    ("2-merge-blocks", "right summary not rescaled", [("d_a * math.exp(m_a - m) + d_b * math.exp(m_b - m)", "d_a * math.exp(m_a - m) + d_b")]),
    ("2-merge-blocks", "scale factor with the wrong sign", [("d_a * math.exp(m_a - m) + d_b * math.exp(m_b - m)", "d_a * math.exp(m - m_a) + d_b * math.exp(m - m_b)")]),
    ("2-merge-blocks", "min instead of max", [("m = max(m_a, m_b)", "m = min(m_a, m_b)")]),
    ("2-merge-blocks", "partial last block dropped", [("range(0, len(x), block_size)", "range(0, len(x) // block_size * block_size, block_size)")]),
    ("2-merge-blocks", "reduction starts from (0, 0)", [("m, d = stats[0]\n    for s in stats[1:]:", "m, d = 0.0, 0.0\n    for s in stats:")]),
    ("3-softmax-topk", "normalizer summed over the kept values only", [
        ("    v = [math.exp(u[i] - m) / d for i in range(k)]", "    v = [math.exp(u[i] - m) / sum(math.exp(t - m) for t in u[:k]) for i in range(k)]")]),
    ("3-softmax-topk", "d not rescaled on a new max", [("d = d * math.exp(m - m_new) + math.exp(x - m_new)", "d = d + math.exp(x - m_new)")]),
    ("3-softmax-topk", "ties: later index wins", [("u[i] < u[i + 1]", "u[i] <= u[i + 1]")]),
    ("3-softmax-topk", "returns ascending order", [("return v, p[:k]", "return v[::-1], p[:k][::-1]")]),
    ("3-softmax-topk", "indices not permuted with values", [("            p[i], p[i + 1] = p[i + 1], p[i]\n", "")]),
    ("3-softmax-topk", "insertion moves the new value up one slot only", [("while i >= 0 and u[i] < u[i + 1]:", "if i >= 0 and u[i] < u[i + 1]:")]),
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
