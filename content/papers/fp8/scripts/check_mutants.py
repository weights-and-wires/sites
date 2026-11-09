#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/fp8/scripts/check_mutants.py
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
    ("1-decode-fp8", "E4M3 reserves the top exponent for specials, as IEEE does (max 240)",
     [('if fmt == "e4m3" and e == (1 << eb) - 1 and m == (1 << mb) - 1:', 'if fmt == "e4m3" and e == (1 << eb) - 1:')]),
    ("1-decode-fp8", "subnormal exponent is -bias instead of 1 - bias",
     [("(m / 2**mb) * 2.0 ** (1 - bias)", "(m / 2**mb) * 2.0 ** (-bias)")]),
    ("1-decode-fp8", "subnormals decoded as normals (implicit 1)", [("if e == 0:", "if False:")]),
    ("1-decode-fp8", "E5M2 infinity decoded as NaN", [('return sign * float("inf") if m == 0 else float("nan")', 'return float("nan")')]),
    ("1-decode-fp8", "sign bit ignored", [('sign = -1.0 if (code >> 7) & 1 else 1.0', "sign = 1.0")]),
    ("1-decode-fp8", "E4M3 given the E5M2 bias", [('"e4m3": {"ebits": 4, "mbits": 3, "bias": 7}', '"e4m3": {"ebits": 4, "mbits": 3, "bias": 15}')]),
    ("2-round-to-fp8", "round half up instead of half to even", [("np.round(a / spacing)", "np.floor(a / spacing + 0.5)")]),
    ("2-round-to-fp8", "truncate instead of round", [("np.round(a / spacing)", "np.floor(a / spacing)")]),
    ("2-round-to-fp8", "no saturation (overflow keeps growing)",
     [('np.minimum(np.round(a / spacing) * spacing, f["max"])', "np.round(a / spacing) * spacing")]),
    ("2-round-to-fp8", "no subnormal clamp (spacing keeps shrinking)", [('e = np.maximum(p - 1, 1 - f["bias"])', "e = p - 1")]),
    ("2-round-to-fp8", "binade exponent off by one (uses p, not p - 1)", [('e = np.maximum(p - 1, 1 - f["bias"])', 'e = np.maximum(p, 1 - f["bias"])')]),
    ("2-round-to-fp8", "sign dropped", [("np.sign(x) * q", "q")]),
    ("2-round-to-fp8", "E5M2 rounded with 3 mantissa bits", [('"e5m2": {"mbits": 2,', '"e5m2": {"mbits": 3,')]),
    ("3-per-tensor-scaling", "scale inverted (amax / max)", [('FORMATS[fmt]["max"] / amax', 'amax / FORMATS[fmt]["max"]')]),
    ("3-per-tensor-scaling", "forgot to unscale after the cast", [("return quantize(x * s, fmt) / s", "return quantize(x * s, fmt)")]),
    ("3-per-tensor-scaling", "signed max instead of max magnitude", [("np.max(np.abs(x))", "np.max(x)")]),
    ("3-per-tensor-scaling", "no guard for an all-zero tensor", [("    if amax == 0.0:\n        return 1.0\n", "")]),
    ("3-per-tensor-scaling", "scale computed but never applied before the cast", [("return quantize(x * s, fmt) / s", "return quantize(x, fmt)")]),
    ("3-per-tensor-scaling", "E4M3 maximum used for every format", [('FORMATS[fmt]["max"] / amax', "448.0 / amax")]),
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
