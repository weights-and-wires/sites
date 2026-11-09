#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/gptq/scripts/check_mutants.py
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
    ("1-rtn-grid", "floor instead of round-to-nearest", [("np.rint(w / scale)", "np.floor(w / scale)")]),
    ("1-rtn-grid", "grid ignores 0 (min and max of the row only)", [
        ("lo = np.minimum(W.min(axis=1), 0.0)", "lo = W.min(axis=1)"),
        ("hi = np.maximum(W.max(axis=1), 0.0)", "hi = W.max(axis=1)")]),
    ("1-rtn-grid", "2**bits steps instead of 2**bits - 1", [("maxq = 2**bits - 1", "maxq = 2**bits")]),
    ("1-rtn-grid", "no clipping to the code range", [("np.clip(np.rint(w / scale) + zero, 0, maxq)", "np.rint(w / scale) + zero")]),
    ("1-rtn-grid", "zero point not rounded", [("zero = np.rint(-lo / scale)", "zero = -lo / scale")]),
    ("1-rtn-grid", "symmetric grid (zero point fixed at the middle)", [("zero = np.rint(-lo / scale)", "zero = np.full_like(scale, (maxq + 1) // 2)")]),
    ("2-error-compensation", "no division by [H^-1]_qq", [("/ Hinv[q, q] * Hinv[:, q]", "* Hinv[:, q]")]),
    ("2-error-compensation", "uses H instead of H^-1", [("Hinv[q, q] * Hinv[:, q]", "Hinv[q, q] * np.linalg.inv(Hinv)[:, q]")]),
    ("2-error-compensation", "wrong sign of the update", [("delta = -(w[q] - wq_hat)", "delta = (w[q] - wq_hat)")]),
    ("2-error-compensation", "spreads the error uniformly", [("* Hinv[:, q]", "* np.ones_like(w) / w.size")]),
    ("2-error-compensation", "uses row q of H^-1 but forgets to fix w[q]", [("return w + delta", "out = w + delta\n    out[q] = w[q]\n    return out")]),
    ("2-error-compensation", "modifies w in place", [("return w + delta", "w += delta\n    return w")]),
    ("3-gptq-cholesky", "no error compensation", [("W[:, j:] -= np.outer(err, U[j, j:])", "pass")]),
    ("3-gptq-cholesky", "forgets to divide by U[j, j]", [("err = (W[:, j] - Q[:, j]) / U[j, j]", "err = (W[:, j] - Q[:, j])")]),
    ("3-gptq-cholesky", "lower factor instead of upper", [("np.linalg.cholesky(np.linalg.inv(H)).T", "np.linalg.cholesky(np.linalg.inv(H))")]),
    ("3-gptq-cholesky", "quantizes from the original weights", [("Q[:, j] = quant(W[:, j], scale, zero, maxq)", "Q[:, j] = quant(W0[:, j], scale, zero, maxq)"),
        ("W = W.copy()", "W = W.copy()\n    W0 = W.copy()")]),
    ("3-gptq-cholesky", "reads a column of U instead of a row", [("np.outer(err, U[j, j:])", "np.outer(err, U[j:, j])")]),
    ("3-gptq-cholesky", "grid recomputed from the updated weights", [("Q[:, j] = quant(W[:, j], scale, zero, maxq)", "Q[:, j] = quant(W[:, j], *make_grid(W, bits))")]),
    ("3-gptq-cholesky", "inverse factor taken from H, not H^-1", [("np.linalg.cholesky(np.linalg.inv(H)).T", "np.linalg.cholesky(H).T")]),
    ("4-lazy-batch", "skips the block update", [("W[:, i2:] -= E @ U[i:i2, i2:]", "pass")]),
    ("4-lazy-batch", "block update uses the wrong rows of U", [("E @ U[i:i2, i2:]", "E @ U[:i2 - i, i2:]")]),
    ("4-lazy-batch", "errors are not reset between blocks (stale E columns)", [
        ("E = np.zeros((d_row, i2 - i))", "E = np.zeros((d_row, blocksize)) if i == 0 else E[:, : i2 - i]")]),
    ("4-lazy-batch", "in-block update touches all later columns", [("W[:, j:i2] -= np.outer(E[:, j - i], U[j, j:i2])", "W[:, j:] -= np.outer(E[:, j - i], U[j, j:])")]),
    ("4-lazy-batch", "last block dropped when d_col is not a multiple of B", [("i2 = min(i + blocksize, d_col)", "i2 = i + blocksize\n        if i2 > d_col:\n            break")]),
    ("4-lazy-batch", "in-block update misses the block's own later columns", [("W[:, j:i2] -= np.outer(E[:, j - i], U[j, j:i2])", "W[:, j:j + 1] -= np.outer(E[:, j - i], U[j, j:j + 1])")]),
    ("4-lazy-batch", "forgets to divide by U[j, j]", [("E[:, j - i] = (W[:, j] - Q[:, j]) / U[j, j]", "E[:, j - i] = (W[:, j] - Q[:, j])")]),
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
