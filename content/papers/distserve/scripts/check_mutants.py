#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/distserve/scripts/check_mutants.py
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
    ("1-prefill-queue", "Equation 1 queue term without the 2", [("return d + rate * d * d / (2 * (1 - rho))", "return d + rate * d * d / (1 - rho)")]),
    ("1-prefill-queue", "Equation 1 stable up to R D = 2", [("    if rho >= 1:\n        return INF\n    return d + rate * d * d / (2", "    if rho >= 2:\n        return INF\n    return d + rate * d * d / (2")]),
    ("1-prefill-queue", "Equation 2 uses the Equation 1 denominator", [("rate * d * d / (4 * (2 - rho))", "rate * d * d / (2 * (1 - rho))")]),
    ("1-prefill-queue", "Equation 2 limit at R D = 1", [("    if rho >= 2:\n        return INF\n    return d + rate * d * d / (4", "    if rho >= 1:\n        return INF\n    return d + rate * d * d / (4")]),
    ("1-prefill-queue", "Equation 3 keeps the full execution time", [("return d / k + rate", "return d + rate")]),
    ("1-prefill-queue", "Equation 3 queue term ignores K", [("rate * d * d / (2 * k * (k - rho))", "rate * d * d / (2 * (k - rho))")]),
    ("1-prefill-queue", "Equation 3 limit at R D = 1", [("    if rho >= k:", "    if rho >= 1:")]),
    ("1-prefill-queue", "better_parallelism is inverted", [('if avg_ttft_inter(rate, d) < avg_ttft_intra(rate, d, k):', 'if avg_ttft_inter(rate, d) > avg_ttft_intra(rate, d, k):')]),
    ("1-prefill-queue", "better_parallelism always picks intra", [('        return "inter"\n', '        return "intra"\n')]),
    ("2-kv-transfer", "keys only, no values", [("return 2 * layers", "return 1 * layers")]),
    ("2-kv-transfer", "bytes per value ignored", [("kv_heads * head_dim * bytes_per_value", "kv_heads * head_dim * 2")]),
    ("2-kv-transfer", "transfer time uses bytes, not bits", [("return n_bytes * 8 / (bandwidth_gbit * 1e9)", "return n_bytes / (bandwidth_gbit * 1e9)")]),
    ("2-kv-transfer", "gigabyte per second instead of gigabit", [("return n_bytes * 8 / (bandwidth_gbit * 1e9)", "return n_bytes / (bandwidth_gbit * 1e9) * 8 / 8")]),
    ("2-kv-transfer", "minimum bandwidth ignores the rate", [("return n_bytes * rate * 8 / 1e9", "return n_bytes * 8 / 1e9")]),
    ("2-kv-transfer", "minimum bandwidth in bytes", [("return n_bytes * rate * 8 / 1e9", "return n_bytes * rate / 1e9")]),
    ("3-goodput-bisection", "attainment counts either SLO", [("if a <= ttft_slo and b <= tpot_slo", "if a <= ttft_slo or b <= tpot_slo")]),
    ("3-goodput-bisection", "attainment uses a strict inequality", [("if a <= ttft_slo and b <= tpot_slo", "if a < ttft_slo and b < tpot_slo")]),
    ("3-goodput-bisection", "attainment ignores TPOT", [("if a <= ttft_slo and b <= tpot_slo", "if a <= ttft_slo")]),
    ("3-goodput-bisection", "attainment divides by the number of good requests", [("return good / len(ttft)", "return good / max(good, 1)")]),
    ("3-goodput-bisection", "bisection returns hi", [("    return lo\n", "    return hi\n")]),
    ("3-goodput-bisection", "bisection moves the wrong end", [("            lo = mid\n        else:\n            hi = mid", "            hi = mid\n        else:\n            lo = mid")]),
    ("3-goodput-bisection", "bisection strict comparison", [("if attainment_at(mid) >= target:", "if attainment_at(mid) > target:")]),
    ("3-goodput-bisection", "bisection ignores lo and hi", [("    for _ in range(iters):\n        mid", "    lo, hi = 0.0, 100.0\n    for _ in range(iters):\n        mid")]),
    ("4-placement-search", "replicas round down", [("math.ceil(rate / goodput - 1e-9)", "math.floor(rate / goodput)")]),
    ("4-placement-search", "replicas without the epsilon", [("math.ceil(rate / goodput - 1e-9)", "math.ceil(rate / goodput)")]),
    ("4-placement-search", "system rate is the faster phase", [("rate = min(n_p * g_prefill, n_d * g_decode)", "rate = max(n_p * g_prefill, n_d * g_decode)")]),
    ("4-placement-search", "system rate is the sum", [("rate = min(n_p * g_prefill, n_d * g_decode)", "rate = n_p * g_prefill + n_d * g_decode")]),
    ("4-placement-search", "budget is exclusive", [("range(1, budget - n_p + 1)", "range(1, budget - n_p)")]),
    ("4-placement-search", "ties keep the larger split", [(" or (abs(rate - best[0]) <= 1e-9 and n_p + n_d < best[1] + best[2])", " or (abs(rate - best[0]) <= 1e-9 and n_p + n_d > best[1] + best[2])")]),
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
