#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/roofline/scripts/check_mutants.py
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
    ("1-roofline", "max instead of min (ignores the roofs as limits)", [("np.minimum(peak, bw *", "np.maximum(peak, bw *")]),
    ("1-roofline", "forgot to multiply bandwidth by intensity", [("np.minimum(peak, bw * np.asarray(intensity, dtype=float))", "np.minimum(peak, np.asarray(intensity, dtype=float))")]),
    ("1-roofline", "ridge point inverted (bw / peak)", [("return peak / bw", "return bw / peak")]),
    ("1-roofline", "ridge point as peak * bw", [("return peak / bw", "return peak * bw")]),
    ("1-roofline", "memory-bound includes the ridge point", [("return bw * intensity < peak", "return bw * intensity <= peak")]),
    ("1-roofline", "regime test flipped", [("return bw * intensity < peak", "return bw * intensity > peak")]),
    ("2-intensity", "matmul FLOPs without the factor 2", [("flops = 2 * m * k * n", "flops = m * k * n")]),
    ("2-intensity", "matmul traffic omits C", [("b * (m * k + k * n + m * n)", "b * (m * k + k * n)")]),
    ("2-intensity", "softmax counts only the read", [("traffic = b * 2 * rows * cols", "traffic = b * rows * cols")]),
    ("2-intensity", "LayerNorm ignores gamma and beta", [("b * (2 * rows * cols + 2 * cols)", "b * (2 * rows * cols)")]),
    ("2-intensity", "bytes per element ignored", [("traffic = b * 2 * rows * cols", "traffic = 4 * 2 * rows * cols")]),
    ("3-kernel-time", "time is the sum of compute and memory time", [('"time": max(t_compute, t_memory)', '"time": t_compute + t_memory')]),
    ("3-kernel-time", "attainable ignores bandwidth", [("min(peak, bw * intensity)", "peak")]),
    ("3-kernel-time", "ridge-point kernel labeled memory-bound", [("t_memory > t_compute", "t_memory >= t_compute")]),
    ("3-kernel-time", "regime labels swapped", [('regime = "memory" if t_memory > t_compute else "compute"', 'regime = "compute" if t_memory > t_compute else "memory"')]),
    ("3-kernel-time", "kernels sorted by intensity", [("    return out", '    return sorted(out, key=lambda r: r["intensity"], reverse=True)')]),
    ("4-tiling", "forgot the C write term", [("(2 * n ** 3 / tile + n ** 2)", "(2 * n ** 3 / tile)")]),
    ("4-tiling", "traffic for only one input matrix", [("2 * n ** 3 / tile + n ** 2", "n ** 3 / tile + n ** 2")]),
    ("4-tiling", "binary search off by one (hi = mid - 1)", [("            hi = mid\n", "            hi = mid - 1\n")]),
    ("4-tiling", "unreachable ridge returns n", [("        return None\n    lo, hi", "        return n\n    lo, hi")]),
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
