#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/mixed-precision/scripts/check_mutants.py
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
    ("1-loss-scaling", "unscale in FP16 (divide before the FP32 cast)",
     [("return stored.astype(np.float32) / s", "return (stored / np.float16(scale)).astype(np.float32)")]),
    ("1-loss-scaling", "forget to unscale", [("return stored.astype(np.float32) / s", "return stored.astype(np.float32)")]),
    ("1-loss-scaling", "scale in FP16 instead of FP32", [("np.asarray(g, dtype=np.float32) * s", "np.asarray(g, dtype=np.float16) * np.float16(scale)")]),
    ("1-loss-scaling", "choose_scale rounds up (ceil)", [("np.floor(", "np.ceil(")]),
    ("1-loss-scaling", "choose_scale returns a non-power-of-two", [("return float(2.0 ** np.floor(np.log2(fp16_max / max_abs_grad)))", "return float(fp16_max / max_abs_grad)")]),
    ("2-master-weights", "update computed in FP16", [("np.float32(lr) * np.asarray(g16, dtype=np.float16).astype(np.float32)", "(np.float16(lr) * np.asarray(g16, dtype=np.float16)).astype(np.float32)")]),
    ("2-master-weights", "master rebuilt from the FP16 copy", [("np.asarray(master, dtype=np.float32) - update", "np.asarray(master, dtype=np.float16).astype(np.float32) - update")]),
    ("2-master-weights", "master stored in FP16", [("(np.asarray(master, dtype=np.float32) - update).astype(np.float32)", "(np.asarray(master, dtype=np.float32) - update).astype(np.float16)")]),
    ("3-fp32-accumulation", "accumulator rounded to FP16 each step",
     [("s = acc_dtype(s + acc_dtype(x) * acc_dtype(y))", "s = np.float16(s + acc_dtype(x) * acc_dtype(y))")]),
    ("3-fp32-accumulation", "FP32 result not converted to FP16", [("return np.float16(s)", "return s")]),
    ("3-fp32-accumulation", "products formed in FP16", [("acc_dtype(x) * acc_dtype(y)", "np.float16(x * y)")]),
    ("4-bfloat16", "truncation instead of rounding", [("bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))", "bias = np.uint32(0)")]),
    ("4-bfloat16", "ties not broken to even", [("bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))", "bias = np.uint32(0x8000)")]),
    ("4-bfloat16", "tie rule inverted (ties to odd)", [("(u >> np.uint32(16)) & np.uint32(1)", "1 - ((u >> np.uint32(16)) & np.uint32(1))")]),
    ("4-bfloat16", "round through FP16 instead", [("return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)", "return x.astype(np.float16).astype(np.float32)")]),
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
    print(f"\n{len(MUTANTS) - survivors} of {len(MUTANTS)} mutants caught, {survivors} survived")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
