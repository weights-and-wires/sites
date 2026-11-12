#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/fa3/scripts/check_mutants.py
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
    ("1-tiled-attention", "no running max (exp of raw scores)", [("m_new = np.maximum(m, S.max(axis=1))", "m_new = np.zeros_like(m)")]),
    ("1-tiled-attention", "max of the current block only", [("m_new = np.maximum(m, S.max(axis=1))", "m_new = S.max(axis=1)")]),
    ("1-tiled-attention", "O not rescaled when the max grows", [("O = alpha[:, None] * O + P @ V[j:j + Bc]", "O = O + P @ V[j:j + Bc]")]),
    ("1-tiled-attention", "l not rescaled when the max grows", [("l = alpha * l + P.sum(axis=1)", "l = l + P.sum(axis=1)")]),
    ("1-tiled-attention", "L returns m + l instead of m + log l", [("m + np.log(l)", "m + l")]),
    ("1-tiled-attention", "divides by l inside the loop", [("        m = m_new\n", "        m = m_new\n        O = O / l[:, None]\n        l = np.ones_like(l)\n")]),
    ("1-tiled-attention", "tau ignored", [("S = tau * Q @ K[j:j + Bc].T", "S = Q @ K[j:j + Bc].T")]),
    ("1-tiled-attention", "drops the last short block", [("range(0, K.shape[0], Bc)", "range(0, K.shape[0] - K.shape[0] % Bc, Bc)")]),
    ("2-pingpong-simulator", "ignores dependencies", [("[end[d] for d in deps]", "[]")]),
    ("2-pingpong-simulator", "ignores resource availability", [("[free.get(resource, 0.0)] + ", "[0.0] + ")]),
    ("2-pingpong-simulator", "busy counts only the first task per resource", [("busy[resource] = busy.get(resource, 0.0) + duration", "busy.setdefault(resource, duration)")]),
    ("2-pingpong-simulator", "utilization divides by total work", [("return busy[resource] / makespan", "return busy[resource] / sum(busy.values())")]),
    ("2-pingpong-simulator", "pingpong: GEMM ignores the previous softmax", [("deps = [] if last_s[wg] is None else [last_s[wg]]", "deps = []")]),
    ("2-pingpong-simulator", "pingpong: B issues before A", [('for wg in ("A", "B"):\n            deps', 'for wg in ("B", "A"):\n            deps')]),
    ("2-pingpong-simulator", "pingpong: only one warpgroup", [('for wg in ("A", "B"):\n            deps', 'for wg in ("A",):\n            deps')]),
    ("2-pingpong-simulator", "pingpong: softmax has no dependency", [("tasks.append((f\"{wg}.S{j}\", \"sfu\", s, [len(tasks) - 1]))", "tasks.append((f\"{wg}.S{j}\", \"sfu\", s, []))")]),
    ("3-block-quantization", "scale is amax, not amax / 448", [("scale = amax / E4M3_MAX", "scale = amax")]),
    ("3-block-quantization", "one scale per tensor", [("amax = np.abs(blk).max()", "amax = np.abs(x).max()")]),
    ("3-block-quantization", "no rescale after rounding", [("round_e4m3(blk / scale) * scale", "round_e4m3(blk / scale)")]),
    ("3-block-quantization", "zero block divides by zero", [("        if amax == 0:\n            continue\n", "")]),
    ("3-block-quantization", "scale per block of columns", [("blk = x[i:i + block_rows]", "blk = x[:, i:i + block_rows]"), ("out[i:i + block_rows] =", "out[:, i:i + block_rows] =")]),
    ("3-block-quantization", "truncates instead of rounding", [("round_e4m3(blk / scale) * scale", "np.trunc(blk / scale) * scale")]),
    ("4-incoherent-processing", "transform not normalized", [("return x / np.sqrt(d)", "return x")]),
    ("4-incoherent-processing", "transform over the first axis", [("d = x.shape[-1]", "x = x.T; d = x.shape[-1]")]),
    ("4-incoherent-processing", "signs applied after the transform", [("return hadamard_transform(x * signs)", "return hadamard_transform(x) * signs")]),
    ("4-incoherent-processing", "signs ignored", [("return hadamard_transform(x * signs)", "return hadamard_transform(x)")]),
    ("4-incoherent-processing", "rotation skipped in the score error", [("Q, K = incoherent(Q, signs), incoherent(K, signs)", "pass")]),
    ("4-incoherent-processing", "only Q rotated", [("Q, K = incoherent(Q, signs), incoherent(K, signs)", "Q = incoherent(Q, signs)")]),
    ("4-incoherent-processing", "error is absolute, not relative", [(" / np.sqrt(np.mean(S ** 2))", "")]),
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
    skip_torch = False
    survivors = 0
    ran = 0
    for entry in MUTANTS:
        folder, name, edits = entry[:3]
        if len(entry) == 4 and skip_torch:
            print(f"SKIP  {folder}: {name} (torch not available)")
            continue
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
