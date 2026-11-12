#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/henderson/scripts/check_mutants.py
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
    ("1-bootstrap-ci", "percentiles of x, not of the resample means", [("np.percentile(means, [", "np.percentile(np.asarray(x, dtype=float), [")]),
    ("1-bootstrap-ci", "resample means taken over the wrong axis", [("].mean(axis=1)", "].mean(axis=0)")]),
    ("1-bootstrap-ci", "alpha used as a percent, not a fraction", [("[100 * alpha / 2, 100 * (1 - alpha / 2)]", "[alpha / 2, 100 - alpha / 2]")]),
    ("1-bootstrap-ci", "tail probability not halved", [("[100 * alpha / 2, 100 * (1 - alpha / 2)]", "[100 * alpha, 100 * (1 - alpha)]")]),
    ("1-bootstrap-ci", "bounds returned in the wrong order", [("return float(lo), float(hi)", "return float(hi), float(lo)")]),
    ("1-bootstrap-ci", "median of the means in place of the upper bound", [("return float(lo), float(hi)", "return float(lo), float(np.median(means))")]),
    ("2-welch-t", "population variance (ddof=0)", [("a.var(ddof=1)", "a.var(ddof=0)"), ("b.var(ddof=1)", "b.var(ddof=0)")]),
    ("2-welch-t", "pooled degrees of freedom", [("df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))", "df = len(a) + len(b) - 2.0")]),
    ("2-welch-t", "df denominator uses n, not n - 1", [("va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)", "va**2 / len(a) + vb**2 / len(b)")]),
    ("2-welch-t", "standard error of the difference not square-rooted", [("(a.mean() - b.mean()) / np.sqrt(va + vb)", "(a.mean() - b.mean()) / (va + vb)")]),
    ("2-welch-t", "variance not divided by n in the statistic", [("va = a.var(ddof=1) / len(a)", "va = a.var(ddof=1)"), ("vb = b.var(ddof=1) / len(b)", "vb = b.var(ddof=1)")]),
    ("2-welch-t", "sign flipped", [("t = (a.mean() - b.mean())", "t = (b.mean() - a.mean())")]),
    ("3-seed-split", "split by even and odd columns", [("a, b = runs[:, :n], runs[:, n:]", "a, b = runs[:, 0::2], runs[:, 1::2]"), ("welch_t_rows(runs[:, :n], runs[:, n:])", "welch_t_rows(runs[:, 0::2], runs[:, 1::2])")]),
    ("3-seed-split", "bands built from the standard deviation", [("a.std(axis=1, ddof=1) / np.sqrt(n)", "a.std(axis=1, ddof=1)"), ("b.std(axis=1, ddof=1) / np.sqrt(n)", "b.std(axis=1, ddof=1)")]),
    ("3-seed-split", "standard error with ddof=0", [("a.std(axis=1, ddof=1)", "a.std(axis=1, ddof=0)"), ("b.std(axis=1, ddof=1)", "b.std(axis=1, ddof=0)")]),
    ("3-seed-split", "band rule uses the larger band only", [("> se_a + se_b", "> np.maximum(se_a, se_b)")]),
    ("3-seed-split", "band rule returns the overlap rate", [("return float(np.mean(np.abs(a.mean(axis=1) - b.mean(axis=1)) > se_a + se_b))", "return float(np.mean(np.abs(a.mean(axis=1) - b.mean(axis=1)) <= se_a + se_b))")]),
    ("3-seed-split", "Welch rule is one-sided", [("np.abs(t) > crit", "t > crit")]),
    ("3-seed-split", "Welch rule compares t, not |t|, to a negated crit", [("np.abs(t) > crit", "t < -crit")]),
    ("4-lift-power", "lift applied to both groups", [("a = x[np.asarray(idx_a)]", "a = lift * x[np.asarray(idx_a)]")]),
    ("4-lift-power", "lift never applied", [("b = lift * x[np.asarray(idx_b)]", "b = x[np.asarray(idx_b)]")]),
    ("4-lift-power", "group B reuses idx_a", [("b = lift * x[np.asarray(idx_b)]", "b = lift * x[np.asarray(idx_a)]")]),
    ("4-lift-power", "one-sided test", [("np.abs(welch_t_rows(a, b)) > crit", "welch_t_rows(a, b) < -crit")]),
    ("4-lift-power", "returns the count, not the fraction", [("return float(np.mean(np.abs(welch_t_rows(a, b)) > crit))", "return float(np.sum(np.abs(welch_t_rows(a, b)) > crit))")]),
    ("4-lift-power", "lift added instead of multiplied", [("b = lift * x[np.asarray(idx_b)]", "b = lift + x[np.asarray(idx_b)]")]),
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
