#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/zero/scripts/check_mutants.py
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
    ("1-memory-model", "stage 1 never shards (threshold off by one)", [("if stage >= 1:\n        opt", "if stage >= 2:\n        opt")]),
    ("1-memory-model", "stage 1 leaves the optimizer state unsharded", [("if stage >= 1:\n        opt = opt / nd", "if stage >= 1:\n        opt = opt")]),
    ("1-memory-model", "stage 3 forgets to shard parameters", [("params = params / nd", "params = params")]),
    ("1-memory-model", "fp32 parameters counted (4 bytes) instead of fp16", [("params = 2 * psi", "params = 4 * psi")]),
    ("1-memory-model", "optimizer multiplier hard-coded to 12", [("opt = K * psi", "opt = 12 * psi")]),
    ("1-memory-model", "max_params ignores nd", [("model_state_bytes(1.0, nd, stage, K)", "model_state_bytes(1.0, 1, stage, K)")]),
    ("2-collectives", "reduce-scatter averages instead of sums", [("xs.sum(axis=0), xs.shape[0])", "xs.mean(axis=0), xs.shape[0])")]),
    ("2-collectives", "reduce-scatter assumes even split", [("np.array_split(xs.sum(axis=0), xs.shape[0])", "list(xs.sum(axis=0).reshape(xs.shape[0], -1))")]),
    ("2-collectives", "all-gather returns each worker only its own shard", [("return [full.copy() for _ in shards]", "return [s.copy() for s in shards]")]),
    ("2-collectives", "stage 3 volume omits the backward all-gather", [("return 1 + 1 + 1", "return 1 + 1")]),
    ("2-collectives", "stage 1 falls through to the stage 3 volume (3 Psi)", [("if stage in (1, 2):\n        return 1 + 1", "if stage in (2,):\n        return 1 + 1")]),
    ("3-zero1-adam", "uses worker 0's gradient, no averaging", [("g = worker_grads.mean(axis=0)", "g = worker_grads[0]")]),
    ("3-zero1-adam", "sums gradients instead of averaging", [("g = worker_grads.mean(axis=0)", "g = worker_grads.sum(axis=0)")]),
    ("3-zero1-adam", "every shard reads the first slice of the gradient", [("g[s], m_shards", "g[:s.stop - s.start], m_shards")]),
    ("3-zero1-adam", "optimizer state is not written back", [("p_i, m_shards[i], v_shards[i] = adam_update(", "p_i, _m, _v = adam_update(")]),
    ("3-zero1-adam", "bias correction stuck at t=1", [("m_shards[i], v_shards[i], t, lr)", "m_shards[i], v_shards[i], 1, lr)")]),
    ("3-zero1-adam", "no all-gather: other slices keep stale parameters", [("return np.concatenate(new)", "return np.concatenate([new[0], params[shard_slices(psi, nd)[1].start:]])")]),
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
