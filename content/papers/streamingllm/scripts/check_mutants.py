#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/streamingllm/scripts/check_mutants.py
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
    ("1-sink-window-cache", "window only (sinks dropped)", [("list(range(sinks)) + list(range(n - window, n))", "list(range(n - window, n))")]),
    ("1-sink-window-cache", "window off by one", [("range(n - window, n)", "range(n - window + 1, n + 1)")]),
    ("1-sink-window-cache", "evicts the oldest token including sinks", [("del out[sinks]", "del out[0]")]),
    ("1-sink-window-cache", "evicts the newest token", [("del out[sinks]", "del out[-1]")]),
    ("1-sink-window-cache", "cache_step mutates its input", [("out = list(cache) + [new_index]", "out = cache; out.append(new_index)")]),
    ("1-sink-window-cache", "evicted set is empty", [("return [i for i in range(n) if i not in kept]", "return []")]),
    ("2-cache-positions", "original indices as positions", [("return np.arange(len(kept))", "return np.asarray(kept)")]),
    ("2-cache-positions", "query rotated at position 0", [("positions[-1:], base)[0]", "positions[:1], base)[0]")]),
    ("2-cache-positions", "keys not rotated", [("K_rot = rotate(K, positions, base)", "K_rot = np.asarray(K, dtype=float)")]),
    ("2-cache-positions", "positions start at 1", [("return np.arange(len(kept))", "return np.arange(1, len(kept) + 1)")]),
    ("2-cache-positions", "query not rotated", [("q_rot = rotate(np.asarray(q, dtype=float)[None, :], positions[-1:], base)[0]", "q_rot = np.asarray(q, dtype=float)")]),
    ("3-sink-softmax", "softmax divides by the max, not the sum", [("e / e.sum()\n\n\ndef softmax_one", "e / e.max()\n\n\ndef softmax_one")]),
    ("3-sink-softmax", "softmax_one is plain softmax", [("return e / (np.exp(-m) + e.sum())", "return e / e.sum()")]),
    ("3-sink-softmax", "softmax_one forgets the shift on the 1", [("return e / (np.exp(-m) + e.sum())", "return e / (1.0 + e.sum())")]),
    ("3-sink-softmax", "softmax_one has no shift (overflows)", [("m = max(np.max(x), 0.0)", "m = 0.0")]),
    ("3-sink-softmax", "attend keeps the old denominator", [("scores = np.asarray(scores, dtype=float)[evict:]", "full = np.asarray(scores, dtype=float)\n    scores = full[evict:]"), ("return softmax(scores) @ V", "return softmax(full)[evict:] @ V")]),
    ("3-sink-softmax", "sink_mass takes the last entries", [("[:num_sinks].sum()", "[-num_sinks:].sum()")]),
    ("3-sink-softmax", "attend ignores evict", [("[evict:]\n    V = np.asarray(V, dtype=float)[evict:]", "[0:]\n    V = np.asarray(V, dtype=float)[0:]")]),
    ("3-sink-softmax", "attend drops scores but not values", [("V = np.asarray(V, dtype=float)[evict:]", "V = np.asarray(V, dtype=float)[:len(scores)]")]),
    ("4-cache-memory", "no factor 2 for keys and values", [("return 2 * layers", "return layers")]),
    ("4-cache-memory", "slots uses max", [("return min(n, sinks + window)", "return max(n, sinks + window)")]),
    ("4-cache-memory", "slots always full", [("return min(n, sinks + window)", "return sinks + window")]),
    ("4-cache-memory", "streaming cache grows", [("return cache_slots(n, sinks, window) * per_token", "return n * per_token")]),
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
