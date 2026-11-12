#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/rnnt/scripts/check_mutants.py
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
    ("1-output-lattice", "softmax over the frame axis", [("probs = h / h.sum(axis=-1, keepdims=True)", "probs = h / h.sum(axis=0, keepdims=True)")]),
    ("1-output-lattice", "no max subtraction (overflows)", [("    logits = logits - logits.max(axis=-1, keepdims=True)\n", "")]),
    ("1-output-lattice", "emit reads the label one step ahead", [("probs[:, u, y[u]]", "probs[:, u, y[u] % (probs.shape[2] - 1) + 1]")]),
    ("1-output-lattice", "emit ignores u (always reads the first label)", [("probs[:, u, y[u]]", "probs[:, u, y[0]]")]),
    ("1-output-lattice", "emit reads the node of the next prefix", [("emit[:, u] = probs[:, u, y[u]]", "emit[:, u] = probs[:, u + 1, y[u]] if u + 1 < probs.shape[1] else 0.0")]),
    ("1-output-lattice", "null read from the first label column", [("null = probs[:, :, 0]", "null = probs[:, :, 1]")]),
    ("1-output-lattice", "g added to f one-to-one instead of by node", [("logits = f[:, None, :] + g[None, :, :]", "logits = f[:, None, :] + g[:1][None, :, :] + 0 * g[None, :, :]")]),
    ("2-forward-brute-force", "forward drops the final null", [("alpha[T - 1, U1 - 1] * null[T - 1, U1 - 1]", "alpha[T - 1, U1 - 1]")]),
    ("2-forward-brute-force", "forward leaves with null[t, u] instead of null[t-1, u]", [("alpha[t - 1, u] * null[t - 1, u]", "alpha[t - 1, u] * null[t, u]")]),
    ("2-forward-brute-force", "forward emit index off by one", [("alpha[t, u - 1] * emit[t, u - 1]", "alpha[t, u - 1] * emit[t, u]")]),
    ("2-forward-brute-force", "forward starts with alpha(1, 0) = null", [("alpha[0, 0] = 1.0", "alpha[0, 0] = null[0, 0]")]),
    ("2-forward-brute-force", "forward ignores the label arrow", [("from_below = alpha[t, u - 1] * emit[t, u - 1] if u > 0 else 0.0", "from_below = alpha[t, u - 1] if u > 0 else 0.0")]),
    ("2-forward-brute-force", "brute force allows an alignment to end on a label", [("itertools.combinations(range(T + U - 1), U)", "itertools.combinations(range(T + U), U)")]),
    ("2-forward-brute-force", "brute force skips the final null", [("        p *= null[T - 1, U]\n", "")]),
    ("2-forward-brute-force", "brute force counts only the first alignment", [("        n += 1\n", "        n = 1\n")]),
    ("3-backward-diagonals", "beta corner is 1", [("beta[t, u] = null[t, u]\n                continue", "beta[t, u] = 1.0\n                continue")]),
    ("3-backward-diagonals", "beta uses null[t+1, u]", [("beta[t + 1, u] * null[t, u]", "beta[t + 1, u] * null[t + 1, u]")]),
    ("3-backward-diagonals", "beta uses emit of the next node", [("beta[t, u + 1] * emit[t, u]", "beta[t, u + 1] * emit[t, u - 1]")]),
    ("3-backward-diagonals", "beta drops the label arrow", [("from_above = beta[t, u + 1] * emit[t, u] if u < U1 - 1 else 0.0", "from_above = beta[t, u + 1] if u < U1 - 1 else 0.0")]),
    ("3-backward-diagonals", "diagonals sum alpha only", [("prod = alpha * beta", "prod = alpha")]),
    ("3-backward-diagonals", "diagonals use rows instead of anti-diagonals", [("out[t0 + u] += prod[t0, u]", "out[min(t0, len(out) - 1)] += prod[t0, u]")]),
    ("3-backward-diagonals", "diagonal index shifted by one", [("out[t0 + u] += prod[t0, u]", "out[(t0 + u + 1) % len(out)] += prod[t0, u]")]),
    ("4-greedy-decode", "no cap on labels per frame", [("for _ in range(max_symbols):", "for _ in range(max_symbols + 40):")]),
    ("4-greedy-decode", "one label at most per frame", [("for _ in range(max_symbols):", "for _ in range(1):")]),
    ("4-greedy-decode", "prefix is not passed to joint", [("joint(t, tuple(out))", "joint(t, ())")]),
    ("4-greedy-decode", "at most one label per frame (breaks after the first label)", [("            k = int(np.argmax(joint(t, tuple(out))))\n            if k == 0:\n                break\n            out.append(k)", "            k = int(np.argmax(joint(t, tuple(out))))\n            if k == 0:\n                break\n            out.append(k)\n            break")]),
    ("4-greedy-decode", "null is emitted as a label", [("            if k == 0:\n                break\n", "")]),
    ("4-greedy-decode", "frame index is wrong", [("joint(t, tuple(out))", "joint(t + 1 if t + 1 < T else t, tuple(out))")]),
    ("4-greedy-decode", "returns a list", [("return tuple(out)", "return out")]),
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
