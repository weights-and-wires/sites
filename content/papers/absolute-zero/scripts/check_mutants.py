#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/absolute-zero/scripts/check_mutants.py
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
    ("1-learnability-reward", "no zero case: a never-solved task earns 1", [("    if mean == 0.0:\n        return 0.0\n", "")]),
    ("1-learnability-reward", "reward is the success rate, not its complement", [("return 1.0 - mean", "return mean")]),
    ("1-learnability-reward", "zero case on mean 1 only", [("if mean == 0.0:", "if mean == 1.0:")]),
    ("2-composite-reward", "well-formed failure earns 0 instead of -0.5", [("return -0.5", "return 0.0")]),
    ("2-composite-reward", "format error earns -0.5", [("return -1.0", "return -0.5")]),
    ("2-composite-reward", "passed is checked before the format", [
        ("    if not well_formed:\n        return -1.0\n    if passed:\n        return float(r_role)\n",
         "    if passed:\n        return float(r_role)\n    if not well_formed:\n        return -1.0\n")]),
    ("2-composite-reward", "role reward replaced by 1", [("return float(r_role)", "return 1.0")]),
    ("3-task-relative-advantages", "baselines per task only, roles pooled", [("keys = list(zip(tasks, roles))", "keys = list(zip(tasks, ['x'] * len(tasks)))")]),
    ("3-task-relative-advantages", "baselines per role only, task types pooled", [("keys = list(zip(tasks, roles))", "keys = list(zip(['x'] * len(roles), roles))")]),
    ("3-task-relative-advantages", "one baseline for the whole batch", [("keys = list(zip(tasks, roles))", "keys = [0] * len(tasks)")]),
    ("3-task-relative-advantages", "sample std (ddof=1)", [("group.std()", "group.std(ddof=1)")]),
    ("3-task-relative-advantages", "no division by the std", [("/ (group.std() + eps)", "")]),
    ("3-task-relative-advantages", "no eps, constant group gives nan", [("(group.std() + eps)", "group.std()")]),
    ("4-check-triplet", "deduction accepts anything", [("return answer == task_output", "return True")]),
    ("4-check-triplet", "abduction compares with the original input", [("return run_program(program_src, answer) == task_output", "return answer == task_input")]),
    ("4-check-triplet", "abduction ignores the program", [("return run_program(program_src, answer) == task_output", "return answer is not None")]),
    ("4-check-triplet", "induction passes on any pair", [("return all(run_program(answer, x) == y for x, y in task_input)", "return any(run_program(answer, x) == y for x, y in task_input)")]),
    ("4-check-triplet", "induction checks only the first pair", [("for x, y in task_input)", "for x, y in task_input[:1])")]),
    ("4-check-triplet", "errors are not caught in abduction", [("        try:\n            return run_program(program_src, answer) == task_output\n        except Exception:\n            return False\n", "        return run_program(program_src, answer) == task_output\n")]),
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
