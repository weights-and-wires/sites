#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/lora/scripts/check_mutants.py
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
    ("1-lora-linear", "B initialized with noise instead of zero", [("B = np.zeros((d, r))", "B = rng.normal(0.0, sigma, size=(d, r))")]),
    ("1-lora-linear", "A initialized to zero (B zero too: no gradient ever)", [("A = rng.normal(0.0, sigma, size=(r, k))", "A = np.zeros((r, k))")]),
    ("1-lora-linear", "A ignores sigma", [("rng.normal(0.0, sigma, size=(r, k))", "rng.normal(0.0, 1.0, size=(r, k))")]),
    ("1-lora-linear", "alpha used without dividing by r", [("(alpha / r) * (X @ A.T)", "alpha * (X @ A.T)")]),
    ("1-lora-linear", "scale multiplies the frozen path too", [("return X @ W0.T + (alpha / r) * (X @ A.T) @ B.T", "return (alpha / r) * (X @ W0.T + (X @ A.T) @ B.T)")]),
    ("1-lora-linear", "alpha times r instead of alpha over r", [("(alpha / r)", "(alpha * r)")]),
    ("1-lora-linear", "B applied before A (uses A^T B^T order)", [("(X @ A.T) @ B.T", "(X @ B) @ A")]),
    ("2-merge-and-swap", "merge forgets alpha / r", [("    return W0 + (alpha / r) * (B @ A)", "    return W0 + (B @ A)")]),
    ("2-merge-and-swap", "unmerge adds instead of subtracts", [("return W - (alpha / r) * (B @ A)", "return W + (alpha / r) * (B @ A)")]),
    ("2-merge-and-swap", "swap forgets to remove the old adapter", [("W0 = unmerge(W, old[0], old[1], alpha)", "W0 = W")]),
    ("2-merge-and-swap", "merge modifies W0 in place", [("return W0 + (alpha / r) * (B @ A)", "W0 += (alpha / r) * (B @ A)\n    return W0")]),
    ("2-merge-and-swap", "scale uses d (B rows) in place of the rank r", [("    r = A.shape[0]\n    return W0 + (alpha / r)", "    r = B.shape[0]\n    return W0 + (alpha / r)")]),
    ("3-lora-gradients", "dA misses the alpha / r scale", [("dA = dZ.T @ X", "dA = (dZ / s).T @ X")]),
    ("3-lora-gradients", "dB misses the alpha / r scale", [("dB = s * dH.T @ Z", "dB = dH.T @ Z")]),
    ("3-lora-gradients", "dA ignores B (uses A as the backward path)", [("dZ = s * dH @ B", "dZ = s * dH @ np.ones_like(B)")]),
    ("3-lora-gradients", "dA built from dB-style outer product (transposed)", [("dZ = s * dH @ B\n    dA = dZ.T @ X", "dZ = s * dH @ B\n    dA = X.T @ dZ\n    dA = dA[: A.shape[0], : A.shape[1]] if dA.shape != A.shape else dA")]),
    ("4-fit-frozen-layer", "B initialized to ones", [("B = np.zeros((d, r))", "B = np.ones((d, r)) * 0.1")]),
    ("4-fit-frozen-layer", "W0 updated in place with the learned delta", [("        A = A - lr * dA\n", "        A = A - lr * dA\n        W0 += 0.0 * lr\n        W0[0, 0] += 1e-6\n")]),
    ("4-fit-frozen-layer", "loss logged after the update, not before", [("        losses.append(float(np.sum(R**2) / (2 * n)))\n", "")]),
    ("4-fit-frozen-layer", "gradient of the mean forgotten (divides by n twice)", [("R / n)", "R / n**2)")]),
    ("4-fit-frozen-layer", "parameter count of a full update", [("return r * (k + d)", "return d * k")]),
    ("4-fit-frozen-layer", "A trained but B never updated", [("        B = B - lr * dB\n", "")]),
    ("4-fit-frozen-layer", "B trained but A never updated", [("        A = A - lr * dA\n", "")]),
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
