#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/mamba/scripts/check_mutants.py
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

ZOH_B = "np.expm1(delta_A) / A[None, :, :] * B[:, None, :]"

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-discretize", "Euler B_bar = delta * B instead of the ZOH formula", [(ZOH_B, "delta[:, :, None] * B[:, None, :]")]),
    ("1-discretize", "ZOH without the division by A", [("np.expm1(delta_A) / A[None, :, :]", "np.expm1(delta_A)")]),
    ("1-discretize", "B_bar = 1 - A_bar (assumes A = -1)", [(ZOH_B, "-np.expm1(delta_A) * B[:, None, :]")]),
    ("1-discretize", "sign flipped: (1 - A_bar) / A", [("np.expm1(delta_A) / A", "-np.expm1(delta_A) / A")]),
    ("1-discretize", "A_bar = exp(A), step size ignored", [("A_bar = np.exp(delta_A)", "A_bar = np.broadcast_to(np.exp(A[None, :, :]), delta_A.shape)")]),
    ("1-discretize", "B_bar ignores the time axis of B", [("* B[:, None, :]", "* B[:1, None, :]")]),
    ("2-selective-scan", "output read from the old state h_{t-1}", [("y_t = (h_new * C_t[None, :]).sum(axis=-1)", "y_t = (h * C_t[None, :]).sum(axis=-1)")]),
    ("2-selective-scan", "C fixed to C_0 (not input-dependent)", [("selective_step(h, x[t], delta[t], A, B[t], C[t])", "selective_step(h, x[t], delta[t], A, B[t], C[0])")]),
    ("2-selective-scan", "delta fixed to delta_0 (time-invariant step)", [("selective_step(h, x[t], delta[t], A, B[t], C[t])", "selective_step(h, x[t], delta[0], A, B[t], C[t])")]),
    ("2-selective-scan", "B fixed to B_0 (not input-dependent)", [("selective_step(h, x[t], delta[t], A, B[t], C[t])", "selective_step(h, x[t], delta[t], A, B[0], C[t])")]),
    ("2-selective-scan", "Euler B_bar inside the recurrence", [(ZOH_B, "delta[:, :, None] * B[:, None, :]")]),
    ("2-selective-scan", "x averaged over channels before the update", [("B_bar[0] * x_t[:, None]", "B_bar[0] * x_t.mean()")]),
    ("2-selective-scan", "state reset to zero every step", [("y_t, h = selective_step(h,", "y_t, h = selective_step(np.zeros_like(h),")]),
    ("2-selective-scan", "output summed over channels instead of state entries", [(".sum(axis=-1)", ".sum(axis=0)")]),
    ("3-parallel-scan", "combine in the wrong order", [("return a2 * a1, a2 * b1 + b2", "return a2 * a1, a1 * b2 + b1")]),
    ("3-parallel-scan", "rounds stop at L // 2", [("while d < L:", "while d < L // 2:")]),
    ("3-parallel-scan", "offset grows by 1 instead of doubling", [("d *= 2", "d += 1")]),
    ("3-parallel-scan", "decay a never updated between rounds", [("a = np.concatenate([a[:d], new_a])", "a = a")]),
    ("3-parallel-scan", "returns the decay instead of the state", [("    return b\n", "    return a\n")]),
    ("4-state-vs-kv-cache", "KV cache counts keys only", [("2 * n_layers * n_heads", "n_layers * n_heads")]),
    ("4-state-vs-kv-cache", "KV cache ignores the element size", [("* seq_len * bytes_per_elem", "* seq_len")]),
    ("4-state-vs-kv-cache", "Mamba state forgets the convolution buffer", [(" + d_inner * (d_conv - 1)", "")]),
    ("4-state-vs-kv-cache", "convolution buffer holds d_conv instead of d_conv - 1 inputs", [("(d_conv - 1)", "d_conv")]),
    ("4-state-vs-kv-cache", "crossover rounds up and lets a tie count", [("state_bytes // per_token + 1", "-(-state_bytes // per_token)")]),
    ("4-state-vs-kv-cache", "crossover is off by one (no + 1)", [("state_bytes // per_token + 1", "state_bytes // per_token")]),
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
