#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/spec-decoding/scripts/check_mutants.py 
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
RES = "residual_distribution(p, q)"
MUTANTS = [
    ("1-speculative-sampling", "resample from p instead of the residual", [("p=residual_distribution(p, q)", "p=p")]),
    ("1-speculative-sampling", "resample from q instead of the residual", [("len(p), p=residual_distribution(p, q)", "len(q), p=q")]),
    ("1-speculative-sampling", "residual uses q - p", [("np.asarray(p, dtype=float) - np.asarray(q, dtype=float)", "np.asarray(q, dtype=float) - np.asarray(p, dtype=float)")]),
    ("1-speculative-sampling", "residual is not normalized", [("return r / r.sum()", "return r")]),
    ("1-speculative-sampling", "always accept the draft token", [("rng.random() <= min(1.0, p[x] / q[x])", "rng.random() <= 1.0")]),
    ("1-speculative-sampling", "ratio inverted (q / p)", [("p[x] / q[x]", "q[x] / p[x]")]),
    ("2-verify-draft", "count rejections instead of stopping at the first", [("            n = i\n            break", "            n = n - 1")]),
    ("2-verify-draft", "residual taken at the previous index", [("residual_distribution(p_dists[n], q_dists[n])", "residual_distribution(p_dists[n - 1], q_dists[n - 1])")]),
    ("2-verify-draft", "no bonus token when every guess is accepted", [("dist = p_dists[gamma]", "dist = p_dists[gamma - 1]")]),
    ("2-verify-draft", "rejection test reversed", [("if r[i] > p_dists[i][x] / q_dists[i][x]:", "if r[i] < p_dists[i][x] / q_dists[i][x]:")]),
    ("2-verify-draft", "keeps one draft token too many", [("draft[:n]]", "draft[:n + 1]]")]),
    ("2-verify-draft", "replacement drawn from p, not the residual", [("dist = residual_distribution(p_dists[n], q_dists[n])", "dist = p_dists[n]")]),
    ("2-verify-draft", "ratio ignores q", [("p_dists[i][x] / q_dists[i][x]", "p_dists[i][x]")]),
    ("3-speedup", "alpha = 1 not handled", [("    if alpha == 1:\n        return float(gamma + 1)\n", "")]),
    ("3-speedup", "the cost of the draft model is ignored", [("/ (gamma * c + 1)", "/ 1")]),
    ("3-speedup", "exponent gamma instead of gamma + 1", [("alpha ** (gamma + 1)", "alpha ** gamma")]),
    ("3-speedup", "alpha uses max instead of min", [("np.minimum(", "np.maximum(")]),
    ("3-speedup", "cost counts gamma + 1 draft runs", [("(gamma * c + 1)", "((gamma + 1) * c + 1)")]),
    ("3-speedup", "ties go to the larger gamma", [("range(1, max_gamma + 1)", "range(max_gamma, 0, -1)")]),
    ("3-speedup", "search range stops one short", [("range(1, max_gamma + 1)", "range(1, max_gamma)")]),
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
