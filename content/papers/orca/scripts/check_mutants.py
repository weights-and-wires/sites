#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/orca/scripts/check_mutants.py
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
    ("1-request-level", "finished requests return early (no wait for the batch)",
     [("        for i in batch:\n            finish[i] = t\n", "        for i in batch:\n            finish[i] = t - steps + lengths[i]\n")]),
    ("1-request-level", "batch also admits requests that arrive mid-batch",
     [("arrivals[order[nxt]] <= t:\n            batch", "True:\n            batch")]),
    ("1-request-level", "useful counts finished requests that sit in the batch",
     [("sum(1 for i in batch if lengths[i] > s)", "len(batch)")]),
    ("1-request-level", "idle gap is not recorded",
     [("useful += [0] * (arrivals[order[nxt]] - t)", "pass")]),
    ("1-request-level", "FCFS by index, not by arrival",
     [("key=lambda i: (arrivals[i], i)", "key=lambda i: i")]),
    ("2-iteration-level", "finish time set after the whole batch (request-level)",
     [("finish[i] = t + 1", "finish[i] = t + 1 + max(remaining[j] for j in ready)")]),
    ("2-iteration-level", "batch is fixed until all its requests finish",
     [("ready = [i for i in order if arrivals[i] <= t and remaining[i] > 0][:max_bs]",
       "ready = [i for i in order if arrivals[i] <= t and remaining[i] > 0][:max_bs]\n        if useful and any(remaining[i] > 0 for i in prev):\n            ready = [i for i in prev if remaining[i] > 0]\n        prev = ready if not useful or not any(remaining[i] > 0 for i in prev) else prev"),
      ("    t = 0\n    done = 0", "    t = 0\n    prev = []\n    done = 0")]),
    ("2-iteration-level", "newly arrived requests wait one extra iteration",
     [("arrivals[i] <= t and", "arrivals[i] < t and")]),
    ("2-iteration-level", "ignores max_bs",
     [("[:max_bs]", "")]),
    ("2-iteration-level", "FCFS by index, not by arrival",
     [("key=lambda i: (arrivals[i], i)", "key=lambda i: i")]),
    ("3-select", "continue instead of break on a failed reservation",
     [("            if new_n_rsrv > n_slots:\n                break", "            if new_n_rsrv > n_slots:\n                continue")]),
    ("3-select", ">= instead of > (exact fit rejected)",
     [("new_n_rsrv > n_slots", "new_n_rsrv >= n_slots")]),
    ("3-select", "INCREMENT requests reserve again",
     [('        if pool[i]["state"] == "INITIATION":\n            new_n_rsrv', '        if True:\n            new_n_rsrv')]),
    ("3-select", "RUNNING requests are not filtered",
     [('if r["state"] != "RUNNING"', "if True")]),
    ("3-select", "not sorted by arrival",
     [("        key=lambda i: pool[i][\"arrival\"],\n", "        key=lambda i: i,\n")]),
    ("3-select", "max_bs ignored",
     [("if len(batch) == max_bs:", "if False:")]),
    ("4-selective-batching", "attention over the flat tensor (requests leak into each other)",
     [("        merged[s] = causal_attention(q[s], k[s], v[s])\n        start += n\n",
       "        start += n\n    merged = causal_attention(q, k, v)\n")]),
    ("4-selective-batching", "start never advances (every request reads request 0's rows)",
     [("        start += n\n", "")]),
    ("4-selective-batching", "forgets the Attn Out Linear",
     [("return merged @ Wo", "return merged")]),
    ("4-selective-batching", "keys and values swapped",
     [("causal_attention(q[s], k[s], v[s])", "causal_attention(q[s], v[s], k[s])")]),
    ("4-selective-batching", "Attn Out Linear applied before attention",
     [("    q, k, v = np.split(x @ Wqkv, 3, axis=1)", "    q, k, v = np.split((x @ Wo) @ Wqkv, 3, axis=1)")]),
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
