#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/flashattention/scripts/check_mutants.py
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
    ("1-tiled-attention", "old output not rescaled when the maximum grows",
     [("(old * l_i)[:, None] * O_i", "l_i[:, None] * O_i")]),
    ("1-tiled-attention", "old normalizer not rescaled",
     [("l_new = old * l_i + new * l_t", "l_new = l_i + new * l_t")]),
    ("1-tiled-attention", "new block not rescaled to the new maximum",
     [("new[:, None] * (P_t @ Vj)", "(P_t @ Vj)")]),
    ("1-tiled-attention", "block-local softmax averaged (no running statistics)",
     [("m_new = np.maximum(m_i, m_t)", "m_new = m_t")]),
    ("1-tiled-attention", "no max subtraction in the block",
     [("P_t = np.exp(S - m_t[:, None])", "P_t = np.exp(S)"),
      ("new = np.exp(m_t - m_new)", "new = np.ones_like(m_t)"),
      ("old = np.exp(m_i - m_new)", "old = np.ones_like(m_i)")]),
    ("1-tiled-attention", "last partial K/V block dropped",
     [("for j in range(0, M, Bc):", "for j in range(0, (M // Bc) * Bc, Bc):")]),
    ("1-tiled-attention", "last partial Q block dropped",
     [("for i in range(0, N, Br):", "for i in range(0, (N // Br) * Br, Br):")]),
    ("1-tiled-attention", "tau ignored",
     [("S = tau * (Q[i:i + Br] @ Kj.T)", "S = Q[i:i + Br] @ Kj.T")]),
    ("1-tiled-attention", "full score matrix materialized",
     [("    for j in range(0, M, Bc):",
       "    _full = np.exp(tau * (Q @ K.T))\n    for j in range(0, M, Bc):")]),
    ("1-tiled-attention", "stored l never updated",
     [("l[i:i + Br], m[i:i + Br] = l_new, m_new", "m[i:i + Br] = m_new")]),
    ("2-count-hbm", "S never written to HBM (fused away in the count)",
     [("line1 = 2 * N * d + N * N", "line1 = 2 * N * d")]),
    ("2-count-hbm", "softmax pass counted once",
     [("line2 = N * N + N * N", "line2 = N * N")]),
    ("2-count-hbm", "l and m traffic forgotten",
     [("loads = N * d + N * d + N + N", "loads = N * d + N * d"), ("writes = N * d + N + N", "writes = N * d")]),
    ("2-count-hbm", "Tc rounded down",
     [("Tc = math.ceil(N / Bc)", "Tc = N // Bc")]),
    ("2-count-hbm", "K and V reloaded on every pass",
     [("return load_kv + Tc * (loads + writes)", "return Tc * (load_kv + loads + writes)")]),
    ("2-count-hbm", "Bc = M / d instead of M / 4d",
     [("Bc = math.ceil(M / (4 * d))", "Bc = math.ceil(M / d)")]),
    ("2-count-hbm", "O only written, not read back each pass",
     [("loads = N * d + N * d + N + N", "loads = N * d + N + N")]),
    ("3-recompute-backward", "P normalized by the block's own row sums",
     [("P = np.exp(S - mi[:, None]) / li[:, None]", "P = np.exp(S - S.max(axis=1, keepdims=True)); P = P / P.sum(axis=1, keepdims=True)")]),
    ("3-recompute-backward", "P not divided by l",
     [("P = np.exp(S - mi[:, None]) / li[:, None]", "P = np.exp(S - mi[:, None])")]),
    ("3-recompute-backward", "softmax Jacobian term D dropped",
     [("dS = P * (dP - D[:, None])", "dS = P * dP")]),
    ("3-recompute-backward", "D computed from dO alone",
     [("D = (dOi * Oi).sum(axis=1)", "D = dOi.sum(axis=1)")]),
    ("3-recompute-backward", "dK uses dS without the transpose",
     [("dS.T @ Qi", "dS @ Qi")]),
    ("3-recompute-backward", "dV uses P without the transpose",
     [("dV_part = P.T @ dOi", "dV_part = P @ dOi")]),
    ("3-recompute-backward", "dQ and dK swapped",
     [("return dS @ Kj, dS.T @ Qi, dV_part", "return dS.T @ Qi, dS @ Kj, dV_part")]),
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
