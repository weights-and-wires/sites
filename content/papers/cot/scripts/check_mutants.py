#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/cot/scripts/check_mutants.py
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
    ("1-chain-verifier", "first answer counts, not the last", [("float(found[-1])", "float(found[0])")]),
    ("1-chain-verifier", "tolerance of 0.5 for every result", [("tol = 1e-9 if decimals == 0 else 0.5 * 10 ** (-decimals) + 1e-9", "tol = 0.5")]),
    ("1-chain-verifier", "exact float comparison", [("return abs(value - c) <= tol", "return value == c")]),
    ("1-chain-verifier", "division by zero accepted", [("        if b == 0:\n            return False", "        if b == 0:\n            return True")]),
    ("1-chain-verifier", "subtraction checked as addition", [('elif op == "-":\n        value = a - b', 'elif op == "-":\n        value = a + b')]),
    ("1-chain-verifier", "a chain with no equations counts as valid", [("sound = steps > 0 and not bad", "sound = not bad")]),
    ("1-chain-verifier", "lucky_correct divided by all traces", [("lucky / correct if correct else 0.0", "lucky / n if n else 0.0")]),
    ("1-chain-verifier", "lucky_correct counts wrong answers too", [("lucky += right and not sound", "lucky += not sound")]),
    ("2-serial-compute-budget", "answer_only ignores the budget", [("for op in ops[:budget]:", "for op in ops:")]),
    ("2-serial-compute-budget", "stepwise draws a slip per operation", [("            value = apply_op(value, op)\n        if rng is not None and rng.random() < eps:\n            value += 1\n        values.append(value)", "            value = apply_op(value, op)\n            if rng is not None and rng.random() < eps:\n                value += 1\n        values.append(value)")]),
    ("2-serial-compute-budget", "a slip is written but not carried forward", [("        if rng is not None and rng.random() < eps:\n            value += 1\n        values.append(value)", "        if rng is not None and rng.random() < eps:\n            values.append(value + 1)\n            continue\n        values.append(value)")]),
    ("2-serial-compute-budget", "the last partial pass is dropped", [("range(0, len(ops), budget)", "range(0, len(ops) - len(ops) % budget, budget)")]),
    ("2-serial-compute-budget", "stepwise answer is the first line", [("values[-1] if values else start", "values[0] if values else start")]),
    ("2-serial-compute-budget", "answer_only never slips", [("    if rng is not None and rng.random() < eps:\n        value += 1\n    return value", "    return value")]),
    ("2-serial-compute-budget", "accuracy truth skips the last operation", [("            for op in ops:\n                truth = apply_op(truth, op)", "            for op in ops[:-1]:\n                truth = apply_op(truth, op)")]),
    ("3-emergence-with-scale", "first positive gain, not the final run", [("        else:\n            start = None", "        else:\n            pass")]),
    ("3-emergence-with-scale", "a gain of zero counts", [("if g > threshold:", "if g >= threshold:")]),
    ("3-emergence-with-scale", "threshold ignored", [("if g > threshold:", "if g > 0:")]),
    ("3-emergence-with-scale", "gain has the wrong sign", [("return [c - s for s, c in zip(standard, cot)]", "return [s - c for s, c in zip(standard, cot)]")]),
    ("3-emergence-with-scale", "size function returns the index", [("sizes[i]", "i")]),
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
    skip_torch = "--skip-torch" in sys.argv or importlib.util.find_spec("torch") is None
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
