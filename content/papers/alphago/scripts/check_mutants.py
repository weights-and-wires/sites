#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/alphago/scripts/check_mutants.py
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
    ("1-puct-select", "bonus drops the square root", [("np.sqrt(N.sum())", "N.sum()")]),
    ("1-puct-select", "square root of this move's visits only", [("np.sqrt(N.sum())", "np.sqrt(N)")]),
    ("1-puct-select", "denominator is N, not 1 + N", [("(1.0 + N)", "np.maximum(N, 1e-9)")]),
    ("1-puct-select", "c_puct is ignored", [("c_puct * P", "P")]),
    ("1-puct-select", "prior is ignored", [("c_puct * P * np.sqrt", "c_puct * np.sqrt")]),
    ("1-puct-select", "illegal moves are not masked", [("np.where(np.asarray(legal, dtype=bool), scores, -np.inf)", "scores")]),
    ("1-puct-select", "argmin instead of argmax", [("np.argmax(scores)", "np.argmin(scores)")]),
    ("1-puct-select", "value term is dropped", [("return Q + c_puct", "return c_puct")]),
    ("2-leaf-and-backup", "lambda weights swapped", [("(1.0 - lam) * v + lam * z", "lam * v + (1.0 - lam) * z")]),
    ("2-leaf-and-backup", "backup never flips the sign", [("sign = -sign", "pass")]),
    ("2-leaf-and-backup", "signs start with + at the leaf edge", [("sign = -1.0\n    for", "sign = 1.0\n    for")]),
    ("2-leaf-and-backup", "visit count not incremented", [("N[e] += 1", "pass")]),
    ("2-leaf-and-backup", "only the last edge is updated", [("for e in reversed(path):", "for e in [path[-1]]:")]),
    ("2-leaf-and-backup", "mixed Q divides by the wrong count", [("Wr / Nr if Nr > 0", "Wr / Nv if Nv > 0")]),
    ("2-leaf-and-backup", "mixed Q ignores lambda", [("(1.0 - lam) * qv + lam * qr", "0.5 * qv + 0.5 * qr")]),
    ("3-nim-search", "backup never flips the sign", [("v = -v\n            tree[state]", "tree[state]")]),
    ("3-nim-search", "terminal leaf scored as a win", [("v = -1.0\n        else", "v = 1.0\n        else")]),
    ("3-nim-search", "value network output not negated for the mover", [("v = value_fn(s)", "v = -value_fn(s)")]),
    ("3-nim-search", "only the last edge is updated", [("for state, a in reversed(path):", "for state, a in path[-1:]:")]),
    ("3-nim-search", "visit counts not recorded", [('tree[state]["N"][a] += 1', "pass")]),
    ("3-nim-search", "returns the value sums, not the visits", [('return tree[n_root]["N"]', 'return tree[n_root]["W"]')]),
    ("4-policy-gradients", "forgets to subtract the softmax", [("return e - p", "return e")]),
    ("4-policy-gradients", "RL update ignores the outcome sign", [("return (z - baseline) * sl_grad(logits, a)", "return abs(z - baseline) * sl_grad(logits, a)")]),
    ("4-policy-gradients", "baseline is ignored", [("(z - baseline)", "z")]),
    ("4-policy-gradients", "baseline is added", [("(z - baseline)", "(z + baseline)")]),
    ("4-policy-gradients", "gradient of the probability, not the log probability", [("return e - p", "return (e - p) * softmax(logits)[a]")]),
    ("4-policy-gradients", "softmax without the max shift", [("np.exp(x - x.max())", "np.exp(x)")]),
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
