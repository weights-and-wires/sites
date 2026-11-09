#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/deepseekmoe/scripts/check_mutants.py
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
    ("1-layout", "shared experts not subtracted from the routed counts", [("n_routed = m * N - Ks", "n_routed = m * N"), ("k_active = m * K - Ks", "k_active = m * K")]),
    ("1-layout", "active count leaves out the shared experts", [("\"active_ffn\": (k_active + Ks) * size", "\"active_ffn\": k_active * size")]),
    ("1-layout", "expert size is 1/N instead of 1/m", [("size = 1.0 / m", "size = 1.0 / N")]),
    ("1-layout", "combinations counts ordered picks", [("comb(n_routed, k_active)", "comb(n_routed, k_active) * 2")]),
    ("2-deepseekmoe-layer", "gates renormalized over the selected experts (Mixtral style)", [(
        "        h[rows] += s[rows, i][:, None] * f(u[rows])",
        "        g = np.take_along_axis(s, idx, axis=-1)\n        g = s[rows, i] / g[rows].sum(axis=-1)\n        h[rows] += g[:, None] * f(u[rows])")]),
    ("2-deepseekmoe-layer", "shared expert down-weighted instead of added with weight 1", [(
        "        h += f(u)", "        h += 0.5 * f(u)")]),
    ("2-deepseekmoe-layer", "residual forgotten", [("h = u.copy()", "h = np.zeros_like(u)")]),
    ("2-deepseekmoe-layer", "softmax over the token axis", [("logits.max(axis=-1, keepdims=True)", "logits.max(axis=0, keepdims=True)"), ("e / e.sum(axis=-1, keepdims=True)", "e / e.sum(axis=0, keepdims=True)")]),
    ("2-deepseekmoe-layer", "every routed expert runs on every token", [(
        "        rows, _ = np.where(idx == i)\n        if rows.size == 0:\n            continue\n        h[rows] += s[rows, i][:, None] * f(u[rows])",
        "        rows = np.arange(u.shape[0])\n        w = np.where((idx == i).any(axis=1), s[:, i], 0.0)\n        h += w[:, None] * f(u)")]),
    ("2-deepseekmoe-layer", "picks the smallest affinities", [("np.argsort(-s,", "np.argsort(s,")]),
    ("3-expert-balance-loss", "f normalized by T only (K' forgotten)", [("f = counts * n / (k * T)", "f = counts * n / T")]),
    ("3-expert-balance-loss", "P taken only from tokens that selected the expert", [("p = s.mean(axis=0)", "p = np.array([s[(idx == i).any(axis=1), i].mean() if (idx == i).any() else 0.0 for i in range(n)])")]),
    ("3-expert-balance-loss", "alpha ignored", [("float(alpha * (f * p).sum())", "float((f * p).sum())")]),
    ("3-expert-balance-loss", "uses f * P mean instead of sum", [("(f * p).sum()", "(f * p).mean()")]),
    ("4-device-balance-loss", "f' summed instead of averaged", [("f[g].mean()", "f[g].sum()")]),
    ("4-device-balance-loss", "P' averaged instead of summed", [("P[g].sum()", "P[g].mean()")]),
    ("4-device-balance-loss", "alpha ignored", [("float(alpha * total)", "float(total)")]),
    ("4-device-balance-loss", "mean divides by the number of experts N', not the group size", [("f[g].mean()", "f[g].sum() / len(f)")]),
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
