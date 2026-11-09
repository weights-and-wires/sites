#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/layernorm/scripts/check_mutants.py
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

FWD = "gain * (x - mu) / np.sqrt(var + eps) + bias"
# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-layernorm-forward", "batch statistics: mean and variance over axis 0",
     [("x.mean(axis=-1, keepdims=True)", "x.mean(axis=0, keepdims=True)"), ("x.var(axis=-1, keepdims=True)", "x.var(axis=0, keepdims=True)")]),
    ("1-layernorm-forward", "mean over features but variance over the batch", [("x.var(axis=-1, keepdims=True)", "x.var(axis=0, keepdims=True)")]),
    ("1-layernorm-forward", "unbiased variance (divide by H - 1)", [("x.var(axis=-1, keepdims=True)", "x.var(axis=-1, keepdims=True, ddof=1)")]),
    ("1-layernorm-forward", "eps added after the square root", [("np.sqrt(var + eps)", "(np.sqrt(var) + eps)")]),
    ("1-layernorm-forward", "no centering (divide by sigma only)", [("(x - mu)", "x")]),
    ("1-layernorm-forward", "gain and bias swapped", [(FWD, "bias * (x - mu) / np.sqrt(var + eps) + gain")]),
    ("1-layernorm-forward", "standard deviation never taken (divide by variance)", [("np.sqrt(var + eps)", "(var + eps)")]),
    ("1-layernorm-forward", "bias dropped", [("+ bias", "")]),
    ("2-layernorm-backward", "only the direct path: dx = g dy / sigma",
     [("dx = (dxhat - dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)) / sigma", "dx = dxhat / sigma")]),
    ("2-layernorm-backward", "missing the variance term", [(" - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)", "")]),
    ("2-layernorm-backward", "missing the mean term", [(" - dxhat.mean(axis=-1, keepdims=True)", "")]),
    ("2-layernorm-backward", "sum instead of mean in the correction terms", [("dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)", "dxhat.sum(axis=-1, keepdims=True) - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)")]),
    ("2-layernorm-backward", "dgain summed over features", [("dgain = (dy * xhat).sum(axis=0)", "dgain = (dy * xhat).sum(axis=-1)")]),
    ("2-layernorm-backward", "dbias not summed over the batch", [("dbias = dy.sum(axis=0)", "dbias = dy")]),
    ("2-layernorm-backward", "dgain uses dy without xhat", [("dgain = (dy * xhat).sum(axis=0)", "dgain = dy.sum(axis=0)")]),
    ("2-layernorm-backward", "gain left out of dxhat", [("dxhat = dy * gain", "dxhat = dy")]),
    ("3-depth-statistics", "normalize once at the input, not at every layer",
     [("        if use_ln:\n            a = layer_norm(a)\n", ""), ("    h = x\n", "    h = layer_norm(x) if use_ln else x\n")]),
    ("3-depth-statistics", "measure after the ReLU", [("out.append(a.std(axis=-1))\n        h = np.maximum(a, 0)", "h = np.maximum(a, 0)\n        out.append(h.std(axis=-1))")]),
    ("3-depth-statistics", "measure before normalizing", [("        if use_ln:\n            a = layer_norm(a)\n        out.append(a.std(axis=-1))", "        out.append(a.std(axis=-1))\n        if use_ln:\n            a = layer_norm(a)")]),
    ("3-depth-statistics", "normalize across the batch axis", [("a = layer_norm(a)", "a = layer_norm(a.T).T")]),
    ("3-depth-statistics", "feed the un-normalized sum to the next layer", [("h = np.maximum(a, 0)", "h = np.maximum(h @ w.T, 0)")]),
    ("3-depth-statistics", "weights applied as W h with a transposed order", [("a = h @ w.T", "a = h @ w")]),
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
