#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/rope/scripts/check_mutants.py
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
    ("1-rope", "half-split pairing (x[:d/2], x[d/2:]) instead of adjacent pairs", [
        ("a, b = x[..., 0::2], x[..., 1::2]", "a, b = x[..., : d // 2], x[..., d // 2 :]"),
        ("out[..., 0::2] = a * cos - b * sin", "out[..., : d // 2] = a * cos - b * sin"),
        ("out[..., 1::2] = a * sin + b * cos", "out[..., d // 2 :] = a * sin + b * cos")]),
    ("1-rope", "clockwise rotation (sign of sin flipped)", [
        ("out[..., 0::2] = a * cos - b * sin", "out[..., 0::2] = a * cos + b * sin"),
        ("out[..., 1::2] = a * sin + b * cos", "out[..., 1::2] = -a * sin + b * cos")]),
    ("1-rope", "theta exponent -j/d instead of -2j/d", [("-2.0 * np.arange(d // 2) / d", "-1.0 * np.arange(d // 2) / d")]),
    ("1-rope", "base argument ignored (hard-coded 10000)", [("theta = base **", "theta = 10000.0 **")]),
    ("1-rope", "same angle for every pair (position times theta_0)", [("base ** (-2.0 * np.arange(d // 2) / d)", "np.ones(d // 2)")]),
    ("1-rope", "positions ignored (every row rotates as position 1)", [("np.asarray(positions, dtype=float)[:, None]", "np.ones((x.shape[-2], 1))")]),
    ("1-rope", "cos and sin swapped", [("cos, sin = np.cos(angle), np.sin(angle)", "sin, cos = np.cos(angle), np.sin(angle)")]),
    ("2-relative-offset", "offset sign flipped: R_(m-n) instead of R_(n-m)", [("rotary_matrix(n - m, d, base)", "rotary_matrix(m - n, d, base)")]),
    ("2-relative-offset", "absolute position of the key only (R_n)", [("rotary_matrix(n - m, d, base)", "rotary_matrix(n, d, base)")]),
    ("2-relative-offset", "rotary matrix with a transposed block", [("[[c, -s], [s, c]]", "[[c, s], [-s, c]]")]),
    ("2-relative-offset", "rotary matrix uses theta = base^(-j/d)", [("-2.0 * j / d", "-1.0 * j / d")]),
    ("2-relative-offset", "rotary matrix is not block diagonal (full 2x2 tiled)", [
        ("R[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[c, -s], [s, c]]", "R[2 * j : 2 * j + 2, :2] = [[c, -s], [s, c]]")]),
    ("3-extend-context", "interpolation scales up instead of down", [("train_len / new_len", "new_len / train_len")]),
    ("3-extend-context", "NTK exponent 1 instead of d/(d-2)", [("s ** (d / (d - 2))", "s")]),
    ("3-extend-context", "NTK divides the base by s^(d/(d-2))", [("base * s ** (d / (d - 2))", "base / s ** (d / (d - 2))")]),
    ("3-extend-context", "angles use j/d instead of 2j/d", [("-2.0 * np.arange(d // 2) / d", "-1.0 * np.arange(d // 2) / d")]),
    ("3-extend-context", "angles have shape (n, d) with repeated columns", [("return np.asarray(positions, dtype=float)[:, None] * theta[None, :]", "return np.repeat(np.asarray(positions, dtype=float)[:, None] * theta[None, :], 2, axis=1)")]),
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
        folder, name, edits = entry[:3]
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
