#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/dit/scripts/check_mutants.py
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
    ("1-patchify", "reshape without the grid transpose", [
        ("z.reshape(g, p, g, p, C).transpose(0, 2, 1, 3, 4).reshape(g * g, p * p * C)", "z.reshape(g * g, p * p * C)")]),
    ("1-patchify", "channels before pixels in each token", [
        ("z.reshape(g, p, g, p, C).transpose(0, 2, 1, 3, 4).reshape(g * g, p * p * C)", "z.reshape(g, p, g, p, C).transpose(0, 2, 4, 1, 3).reshape(g * g, p * p * C)")]),
    ("1-patchify", "column-major patch order", [
        ("z.reshape(g, p, g, p, C).transpose(0, 2, 1, 3, 4).reshape(g * g, p * p * C)", "z.reshape(g, p, g, p, C).transpose(2, 0, 1, 3, 4).reshape(g * g, p * p * C)")]),
    ("1-patchify", "unpatchify skips the transpose", [
        ("tokens.reshape(g, g, p, p, C).transpose(0, 2, 1, 3, 4).reshape(g * p, g * p, C)", "tokens.reshape(g * p, g * p, C)")]),
    ("2-adaln-zero-block", "scale without the 1 + (zero init breaks)", [("h * (1 + scale) + shift", "h * scale + shift")]),
    ("2-adaln-zero-block", "gate applied to the sum, not the branch", [
        ("x = x + gate1 * attn(modulate(layer_norm(x), shift1, scale1))", "x = gate1 * (x + attn(modulate(layer_norm(x), shift1, scale1)))")]),
    ("2-adaln-zero-block", "no gate on the MLP half", [("x = x + gate2 * mlp(", "x = x + mlp(")]),
    ("2-adaln-zero-block", "shift and scale swapped", [
        ("shift1, scale1, gate1, shift2, scale2, gate2 =", "scale1, shift1, gate1, scale2, shift2, gate2 =")]),
    ("2-adaln-zero-block", "MLP half reuses the attention modulation", [
        ("modulate(layer_norm(x), shift2, scale2)", "modulate(layer_norm(x), shift1, scale1)")]),
    ("3-gflops", "adaLN layer charged per token", [("6 * d * d", "6 * T * d * d")]),
    ("3-gflops", "attention score term linear in T", [("2 * T * T * d", "2 * T * d")]),
    ("3-gflops", "MLP expansion 2 instead of 4", [("8 * T * d * d", "4 * T * d * d")]),
    ("3-gflops", "tokens counted as I // p instead of (I // p) ** 2", [("T = (I // p) ** 2", "T = I // p")]),
    ("3-gflops", "decoder outputs C channels, not 2C", [("p * p * 2 * C", "p * p * C")]),
    ("4-guidance-channels", "guide all four noise channels", [("out[:n_guided] =", "out[:C] ="), ("out_uncond[:n_guided] + s * (out_cond[:n_guided] - out_uncond[:n_guided])", "out_uncond[:C] + s * (out_cond[:C] - out_uncond[:C])")]),
    ("4-guidance-channels", "guide the covariance channels too", [("out = out_cond.copy()", "out = out_uncond + s * (out_cond - out_uncond)")]),
    ("4-guidance-channels", "scale applied to the unconditional term", [("out_uncond[:n_guided] + s * (out_cond[:n_guided] - out_uncond[:n_guided])", "s * out_uncond[:n_guided] + (out_cond[:n_guided] - out_uncond[:n_guided])")]),
    ("4-guidance-channels", "modify the input in place", [("out = out_cond.copy()", "out = out_cond")]),
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
