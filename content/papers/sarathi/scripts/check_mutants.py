#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/sarathi/scripts/check_mutants.py
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
    ("1-roofline-iteration", "intensity ignores the 2 FLOPs per multiply-add", [("return 2.0 * n_tokens / bytes_per_param", "return 1.0 * n_tokens / bytes_per_param")]),
    ("1-roofline-iteration", "intensity ignores bytes per parameter", [("return 2.0 * n_tokens / bytes_per_param", "return 2.0 * n_tokens / 2")]),
    ("1-roofline-iteration", "sum instead of max", [("return max(t_math, t_mem), (", "return t_math + t_mem, (")]),
    ("1-roofline-iteration", "memory time scales with tokens", [("t_mem = n_params * bytes_per_param / bandwidth\n    return max", "t_mem = n_params * bytes_per_param * n_tokens / bandwidth\n    return max")]),
    ("1-roofline-iteration", "bound labels swapped", [('("memory" if t_mem > t_math else "compute")', '("compute" if t_mem > t_math else "memory")')]),
    ("1-roofline-iteration", "budget ignores the memory floor", [("    if slo < n_params * bytes_per_param / bandwidth:\n        return 0\n", "")]),
    ("1-roofline-iteration", "budget off by one (ceil)", [("math.floor(slo * peak_flops / (2.0 * n_params) + 1e-9)", "math.ceil(slo * peak_flops / (2.0 * n_params) + 1e-9)")]),
    ("2-chunk-the-prompt", "extra empty chunk on exact multiples", [("    return sizes\n", "    return sizes + [0] if prompt_len % budget == 0 else sizes\n")]),
    ("2-chunk-the-prompt", "last chunk padded to the budget", [("c = min(budget, left)", "c = budget")]),
    ("2-chunk-the-prompt", "re-read counts the chunk itself", [("sum(chunks[:i]) for i in range(len(chunks))", "sum(chunks[:i + 1]) for i in range(len(chunks))")]),
    ("2-chunk-the-prompt", "re-read counts only the previous chunk", [("sum(chunks[:i]) for i in range(len(chunks))", "chunks[i - 1] for i in range(1, len(chunks))")]),
    ("3-stall-free-batch", "decodes not subtracted from the budget", [("left = max(0, budget - n_decodes)", "left = budget")]),
    ("3-stall-free-batch", "left can go negative", [("left = max(0, budget - n_decodes)", "left = budget - n_decodes")]),
    ("3-stall-free-batch", "chunk not capped by remaining", [("c = min(remaining, left)", "c = left")]),
    ("3-stall-free-batch", "shortest request first", [("for remaining in waiting:", "for remaining in sorted(waiting):")]),
    ("3-stall-free-batch", "no left decrement", [("        left -= c\n", "")]),
    ("4-stall-simulation", "stall-free ignores decodes when sizing the chunk", [("max(0, budget - len(active))", "budget")]),
    ("4-stall-simulation", "stall-free skips decodes in a mixed iteration", [("n, include = len(active) + chunk, True", "n, include = len(active) + chunk, chunk == 0")]),
    ("4-stall-simulation", "prefill-first lets decodes emit during the prefill", [("n, include, chunk = remaining, False, remaining", "n, include, chunk = remaining, True, remaining")]),
    ("4-stall-simulation", "TTFT measured from time zero", [("ttft = t - t_arrive", "ttft = t")]),
    ("4-stall-simulation", "TBT is the longest iteration, not the longest gap", [("gaps = [b - a for ts in times for a, b in zip(ts, ts[1:])]", "gaps = [iter_time(budget)]")]),
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
