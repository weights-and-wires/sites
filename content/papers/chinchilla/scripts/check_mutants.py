#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/chinchilla/scripts/check_mutants.py
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
    ("1-parametric-loss", "data term uses alpha instead of beta",
     [("B / d**BETA\n\n\ndef isoflop", "B / d**ALPHA\n\n\ndef isoflop")]),
    ("1-parametric-loss", "A and B swapped",
     [("E + A / n**ALPHA + B / d**BETA\n\n\ndef isoflop", "E + B / n**ALPHA + A / d**BETA\n\n\ndef isoflop")]),
    ("1-parametric-loss", "irreducible loss E dropped",
     [("return E + A / n**ALPHA", "return A / n**ALPHA")]),
    ("1-parametric-loss", "tokens from C = 2ND (forward pass only)",
     [("d = c / (6 * sizes)", "d = c / (2 * sizes)")]),
    ("1-parametric-loss", "tokens from C = ND",
     [("d = c / (6 * sizes)", "d = c / sizes")]),
    ("1-parametric-loss", "fixed 20 tokens per parameter instead of the budget",
     [("d = c / (6 * sizes)", "d = 20 * sizes")]),
    ("2-efficient-frontier", "a and b swapped (Kaplan-style direction)",
     [("a = BETA / (ALPHA + BETA)\n    b = ALPHA / (ALPHA + BETA)", "a = ALPHA / (ALPHA + BETA)\n    b = BETA / (ALPHA + BETA)")]),
    ("2-efficient-frontier", "G built from beta A / alpha B (inverted)",
     [("(ALPHA * A / (BETA * B))", "(BETA * B / (ALPHA * A))")]),
    ("2-efficient-frontier", "power laws of C instead of C / 6",
     [("g * (c / 6) ** a, (c / 6) ** b / g", "g * c ** a, c ** b / g")]),
    ("2-efficient-frontier", "G exponent 1 instead of 1 / (alpha + beta)",
     [("** (1 / (ALPHA + BETA))", "** 1")]),
    ("2-efficient-frontier", "D multiplied by G instead of divided",
     [("(c / 6) ** b / g", "(c / 6) ** b * g")]),
    ("3-isoflop-parabola", "parabola in N, not ln N",
     [("x = np.log(np.asarray(sizes, dtype=float))", "x = np.asarray(sizes, dtype=float)")]),
    ("3-isoflop-parabola", "vertex without the factor 2",
     [("x_star = -c1 / (2 * c2)", "x_star = -c1 / c2")]),
    ("3-isoflop-parabola", "returns ln N and not N",
     [("float(np.exp(x_star))", "float(x_star)")]),
    ("3-isoflop-parabola", "returns the loss of the lowest sampled point",
     [("float(np.polyval([c2, c1, c0], x_star))", "float(np.min(losses))")]),
    ("3-isoflop-parabola", "fits a line, not a parabola",
     [("np.polyfit(x, np.asarray(losses, dtype=float), 2)\n    x_star", "(0.0, *np.polyfit(x, np.asarray(losses, dtype=float), 1))\n    x_star")]),
    ("4-fit-exponent", "log10 for the data and natural log for the sizes",
     [("np.log(np.asarray(y, dtype=float))", "np.log10(np.asarray(y, dtype=float))")]),
    ("4-fit-exponent", "fits C against y (inverted regression)",
     [("np.polyfit(np.log(np.asarray(c, dtype=float)), np.log(np.asarray(y, dtype=float)), 1)", "np.polyfit(np.log(np.asarray(y, dtype=float)), np.log(np.asarray(c, dtype=float)), 1)")]),
    ("4-fit-exponent", "fit on raw y against ln c",
     [("np.log(np.asarray(y, dtype=float))", "np.asarray(y, dtype=float)")]),
    ("4-fit-exponent", "tokens grow with the same exponent as the model",
     [("d0 * k ** (1 - a)", "d0 * k**a")]),
    ("4-fit-exponent", "tokens exponent 1 - a applied to the model",
     [("n0 * k**a, d0 * k ** (1 - a)", "n0 * k ** (1 - a), d0 * k**a")]),
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
