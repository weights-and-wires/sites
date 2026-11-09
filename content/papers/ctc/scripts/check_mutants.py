#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/ctc/scripts/check_mutants.py
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
    ("1-collapse-brute-force", "blanks dropped before repeats merge (a-a becomes a)",
     [("    for s in path:\n        if s != prev", "    for s in [x for x in path if x != blank]:\n        if s != prev")]),
    ("1-collapse-brute-force", "blank does not reset the previous symbol",
     [("        prev = s\n    return tuple(out)", "        if s != blank:\n            prev = s\n    return tuple(out)")]),
    ("1-collapse-brute-force", "blank index hard-coded to 0", [("s != blank", "s != 0")]),
    ("1-collapse-brute-force", "max over paths instead of the sum",
     [("total += float(np.prod([y[t, k] for t, k in enumerate(path)]))", "total = max(total, float(np.prod([y[t, k] for t, k in enumerate(path)])))")]),
    ("2-forward-algorithm", "skip allowed between equal labels", [("lp[s] != blank and lp[s] != lp[s - 2]", "lp[s] != blank")]),
    ("2-forward-algorithm", "no skip transition at all", [("if s >= 2 and lp[s] != blank and lp[s] != lp[s - 2]:", "if False:")]),
    ("2-forward-algorithm", "p ignores the final blank state (Equation 8)", [("alpha[T - 1, S - 1] + (alpha[T - 1, S - 2] if S > 1 else 0.0)", "alpha[T - 1, S - 2] if S > 1 else alpha[T - 1, 0]")]),
    ("2-forward-algorithm", "p ignores the last label state", [("alpha[T - 1, S - 1] + (alpha[T - 1, S - 2] if S > 1 else 0.0)", "alpha[T - 1, S - 1]")]),
    ("2-forward-algorithm", "initialization omits the first label", [("alpha[0, 1] = y[0, lp[1]]", "alpha[0, 1] = 0.0")]),
    ("2-forward-algorithm", "missing the s-1 transition", [("total += alpha[t - 1, s - 1]", "pass")]),
    ("2-forward-algorithm", "forgets to multiply by y", [("alpha[t, s] = total * y[t, lp[s]]", "alpha[t, s] = total")]),
    ("3-best-path-decoding", "arg max over frames instead of symbols", [("np.argmax(y, axis=1)", "np.argmax(y, axis=0)")]),
    ("3-best-path-decoding", "blanks are not removed", [("if s != prev and s != blank:", "if s != prev:")]),
    ("3-best-path-decoding", "blanks dropped before repeats merge",
     [("    for s in path:\n        if s != prev", "    for s in [x for x in path if x != blank]:\n        if s != prev")]),
    ("3-best-path-decoding", "path probability sums instead of multiplying", [("np.prod(", "np.sum(")]),
    ("4-ctc-gradient", "backward skip ignores the repeat rule", [("lp[s] != blank and lp[s] != lp[s + 2]", "lp[s] != blank")]),
    ("4-ctc-gradient", "backward omits the last label state", [("beta[T - 1, S - 2] = y[T - 1, lp[S - 2]]", "pass")]),
    ("4-ctc-gradient", "gradient divides by y squared (derivative w.r.t. y, not u)", [("acc / (y[t] * p)", "acc / (y[t] ** 2 * p)")]),
    ("4-ctc-gradient", "gradient sign flipped", [("grad[t] = y[t] - acc / (y[t] * p)", "grad[t] = acc / (y[t] * p) - y[t]")]),
    ("4-ctc-gradient", "gradient omits the normalization by p", [("acc / (y[t] * p)", "acc / y[t]")]),
    ("4-ctc-gradient", "repeated symbols overwrite instead of accumulating", [("acc[k] += alpha[t, s] * beta[t, s]", "acc[k] = alpha[t, s] * beta[t, s]")]),
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
