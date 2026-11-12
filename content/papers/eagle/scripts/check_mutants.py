#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/eagle/scripts/check_mutants.py 
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
    ("1-tree-mask", "positions use depth, not ctx_len + depth - 1", [("ctx_len + tree_depths(parents) - 1", "tree_depths(parents)")]),
    ("1-tree-mask", "positions are off by one", [("ctx_len + tree_depths(parents) - 1", "ctx_len + tree_depths(parents)")]),
    ("1-tree-mask", "positions are the node index (flattened tree)", [("ctx_len + tree_depths(parents) - 1", "ctx_len + np.arange(len(parents))")]),
    ("1-tree-mask", "ancestors are not visible", [("            j = parents[j]\n    return mask", "            j = -1\n    return mask")]),
    ("1-tree-mask", "context is hidden", [("mask[:, :ctx_len] = True", "mask[:, :ctx_len] = False")]),
    ("1-tree-mask", "causal mask over tree indices (siblings visible)", [("            j = parents[j]\n    return mask", "            j = parents[j]\n    mask[:, ctx_len:] = np.tril(np.ones((n, n), dtype=bool))\n    return mask")]),
    ("1-tree-mask", "node cannot see itself", [("        j = i\n        while j >= 0:", "        j = parents[i]\n        while j >= 0:")]),
    ("2-multi-candidate", "target is not updated after a rejection", [("        cur = residual_distribution(cur, q)\n", "")]),
    ("2-multi-candidate", "final token is drawn from the original p", [("rng.choice(len(cur), p=cur)", "rng.choice(len(p), p=p)")]),
    ("2-multi-candidate", "final token is drawn from q", [("rng.choice(len(cur), p=cur)", "rng.choice(len(q), p=q)")]),
    ("2-multi-candidate", "residual of p, not of the current target", [("cur = residual_distribution(cur, q)", "cur = residual_distribution(p, q)")]),
    ("2-multi-candidate", "only the first candidate is tested", [("zip(cands, u)", "zip(cands[:1], u[:1])")]),
    ("2-multi-candidate", "ratio is inverted", [("cur[x] / q[x]", "q[x] / cur[x]")]),
    ("2-multi-candidate", "accepted index is not reported", [("return x, i", "return x, 0")]),
    ("3-expected-length", "bonus token is dropped", [("1.0 + total", "total")]),
    ("3-expected-length", "siblings are tested independently", [("        remaining[p] *= 1.0 - accept[i]\n", "")]),
    ("3-expected-length", "child ignores its parent's acceptance", [("1.0 if p < 0 else reach[p]", "1.0")]),
    ("3-expected-length", "every sibling is tested after the first accepts", [("remaining[p] *= 1.0 - accept[i]", "remaining[p] *= accept[i]")]),
    ("3-expected-length", "accept probability applied twice", [("reach[i] = remaining[p] * accept[i]", "reach[i] = remaining[p] * accept[i] * accept[i]")]),
    ("4-drafter-step", "token is not advanced", [("E)[tokens[1:]]", "E)[tokens[:-1]]")]),
    ("4-drafter-step", "embedding first, then feature", [("[F[:-1], np.asarray(E)[tokens[1:]]]", "[np.asarray(E)[tokens[1:]], F[:-1]]")]),
    ("4-drafter-step", "bias is dropped", [("@ W + b", "@ W")]),
    ("4-drafter-step", "loss is L1 only", [("np.where(x < beta, 0.5 * x**2 / beta, x - 0.5 * beta)", "x")]),
    ("4-drafter-step", "loss is L2 only", [("np.where(x < beta, 0.5 * x**2 / beta, x - 0.5 * beta)", "0.5 * x**2")]),
    ("4-drafter-step", "linear branch forgets the 0.5 * beta offset", [("x - 0.5 * beta", "x")]),
    ("4-drafter-step", "loss is summed, not averaged", [("beta).mean())", "beta).sum())")]),
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
