#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/fa2/scripts/check_mutants.py
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
    ("1-fa2-forward", "old unscaled output not rescaled when the maximum grows",
     [("O_t = alpha[:, None] * O_t + P @ V[j:j + Bc]", "O_t = O_t + P @ V[j:j + Bc]")]),
    ("1-fa2-forward", "old normalizer not rescaled",
     [("l = alpha * l + P.sum(axis=1)", "l = l + P.sum(axis=1)")]),
    ("1-fa2-forward", "final divide by l forgotten",
     [("O[i:i + Br] = O_t / l[:, None]", "O[i:i + Br] = O_t")]),
    ("1-fa2-forward", "divides by the last block's row sum instead of the running l",
     [("l = alpha * l + P.sum(axis=1)", "l = P.sum(axis=1)")]),
    ("1-fa2-forward", "stores m instead of the logsumexp",
     [("L[i:i + Br] = m + np.log(l)", "L[i:i + Br] = m")]),
    ("1-fa2-forward", "stores log(l) without m",
     [("L[i:i + Br] = m + np.log(l)", "L[i:i + Br] = np.log(l)")]),
    ("1-fa2-forward", "no running maximum, block-local maximum only",
     [("m_new = np.maximum(m, S.max(axis=1))", "m_new = S.max(axis=1)")]),
    ("1-fa2-forward", "no max subtraction",
     [("P = np.exp(S - m_new[:, None])", "P = np.exp(S)"),
      ("alpha = np.exp(m - m_new)", "alpha = np.ones_like(m)")]),
    ("1-fa2-forward", "last partial K/V block dropped",
     [("for j in range(0, M, Bc):", "for j in range(0, (M // Bc) * Bc, Bc):")]),
    ("1-fa2-forward", "last partial Q block dropped",
     [("for i in range(0, N, Br):", "for i in range(0, (N // Br) * Br, Br):")]),
    ("1-fa2-forward", "tau ignored",
     [("S = tau * (Qi @ K[j:j + Bc].T)", "S = Qi @ K[j:j + Bc].T")]),
    ("1-fa2-forward", "full score matrix materialized",
     [("    for i in range(0, N, Br):", "    _full = np.exp(tau * (Q @ K.T))\n    for i in range(0, N, Br):")]),
    ("2-count-nonmatmul", "FA2 final scale charged on every tile",
     [("return Tr * (Tc * per_tile + final)", "return Tr * Tc * (per_tile + final)")]),
    ("2-count-nonmatmul", "FA2 final scale forgotten",
     [("return Tr * (Tc * per_tile + final)", "return Tr * Tc * per_tile")]),
    ("2-count-nonmatmul", "tile counts rounded down",
     [("Tr = math.ceil(N / Br)", "Tr = N // Br"), ("Tc = math.ceil(N / Bc)", "Tc = N // Bc")]),
    ("2-count-nonmatmul", "FA1 charged one scaling of O instead of three",
     [("per_tile = shared + 9 * Br + 3 * Br * d", "per_tile = shared + 9 * Br + Br * d")]),
    ("2-count-nonmatmul", "matmul FLOPs counted as 2N^2 d",
     [("return 4 * N * N * d", "return 2 * N * N * d")]),
    ("2-count-nonmatmul", "time share ignores the cost ratio",
     [("nm = ratio * nonmatmul_ops(N, d, Br, Bc, version)", "nm = nonmatmul_ops(N, d, Br, Bc, version)")]),
    ("3-thread-block-grid", "FA2 grid forgets to round up the row blocks",
     [("blocks *= math.ceil(N / Br)", "blocks *= N // Br")]),
    ("3-thread-block-grid", "FA1 grid multiplied by row blocks",
     [("    if parallel_seq:\n        blocks *= math.ceil(N / Br)", "    blocks *= math.ceil(N / Br)")]),
    ("3-thread-block-grid", "backward grid uses row blocks only (batch and heads dropped)",
     [("return batch * heads * math.ceil(N / Bc)", "return math.ceil(N / Bc)")]),
    ("3-thread-block-grid", "waves rounded down",
     [("return math.ceil(blocks / sms)", "return max(1, blocks // sms)")]),
    ("3-thread-block-grid", "utilization ignores the idle tail of the last wave",
     [("return blocks / (waves(blocks, sms) * sms)", "return min(1.0, blocks / sms)")]),
    ("4-causal-tiles", "skip test uses >= so diagonal-touching tiles are skipped",
     [("if col_min > row_max:", "if col_min >= row_max:")]),
    ("4-causal-tiles", "every computed tile counted as masked",
     [("elif col_max > row_min:", "elif True:")]),
    ("4-causal-tiles", "mask test compares tile origins only",
     [("elif col_max > row_min:", "elif j == i:")]),
    ("4-causal-tiles", "last short block ignored",
     [("for i in range(0, N, Br):", "for i in range(0, (N // Br) * Br, Br):")]),
    ("4-causal-tiles", "speedup bound uses computed over total",
     [("return total / (masked + full)", "return (masked + full) / total")]),
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
