#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/soundstream/scripts/check_mutants.py [--skip-torch]
"""

import importlib.util
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
    ("1-rvq", "quantize y at every stage instead of the residual", [("d = ((residual[:, None, :] - c[None]) ** 2).sum(-1)", "d = ((y[:, None, :] - c[None]) ** 2).sum(-1)")]),
    ("1-rvq", "never subtract the chosen vector from the residual", [("        residual = residual - q\n", "")]),
    ("1-rvq", "largest dot product instead of nearest distance", [("d = ((residual[:, None, :] - c[None]) ** 2).sum(-1)\n        idx = d.argmin(axis=1)", "d = residual @ c.T\n        idx = d.argmax(axis=1)")]),
    ("1-rvq", "decode uses only the first stage", [("for i in range(indices.shape[1]):", "for i in range(1):")]),
    ("1-rvq", "decode ignores the number of index columns (uses all codebooks)", [("for i in range(indices.shape[1]):\n        out = out + codebooks[i][indices[:, i]]", "for i in range(len(codebooks)):\n        out = out + codebooks[i][indices[:, min(i, indices.shape[1] - 1)]]")]),
    ("2-error-vs-codebooks", "fit every stage on X instead of the residual", [("c = kmeans(residual, n_codes, seed=seed + i)", "c = kmeans(np.asarray(X, dtype=float), n_codes, seed=seed + i)")]),
    ("2-error-vs-codebooks", "evaluate each stage on X, not the running residual", [("x_hat = x_hat + quantize(c, X - x_hat)", "x_hat = quantize(c, X)")]),
    ("2-error-vs-codebooks", "entry n uses only codebook n", [("x_hat = x_hat + quantize(c, X - x_hat)", "x_hat = quantize(c, X - x_hat)")]),
    ("2-error-vs-codebooks", "entry 0 is zero instead of the signal power", [("out = [float(np.mean(X ** 2))]", "out = [0.0]")]),
    ("3-bitrate-and-dropout", "sum of strides instead of product", [("fs / math.prod(strides)", "fs / sum(strides)")]),
    ("3-bitrate-and-dropout", "n_codes instead of log2(n_codes) bits", [("n_q * math.log2(n_codes)", "n_q * n_codes")]),
    ("3-bitrate-and-dropout", "round up the quantizer count", [("math.floor(target_bps / per_quantizer + 1e-9)", "math.ceil(target_bps / per_quantizer - 1e-9)")]),
    ("3-bitrate-and-dropout", "dropout never picks the maximum", [("rng.integers(1, n_q_max + 1)", "rng.integers(1, n_q_max)")]),
    ("3-bitrate-and-dropout", "dropout can pick zero quantizers", [("rng.integers(1, n_q_max + 1)", "rng.integers(0, n_q_max + 1)")]),
    ("4-hinge-losses", "one flat mean over all logits", [("return float(np.mean([np.mean(v) for v in per_k]))", "return float(np.concatenate([np.ravel(v) for v in per_k]).mean())")]),
    ("4-hinge-losses", "no hinge clipping on the fake term", [("1.0 + np.asarray(d, dtype=float)) for d in fake_logits])\n    return real + fake", "1.0 + np.asarray(d, dtype=float)) + np.minimum(0.0, 1.0 + np.asarray(d, dtype=float)) for d in fake_logits])\n    return real + fake")]),
    ("4-hinge-losses", "fake term sign flipped (1 - D(G(x)))", [("np.maximum(0.0, 1.0 + np.asarray(d, dtype=float)) for d in fake_logits])\n    return real + fake", "np.maximum(0.0, 1.0 - np.asarray(d, dtype=float)) for d in fake_logits])\n    return real + fake")]),
    ("4-hinge-losses", "generator uses the discriminator's fake term (1 + D(G(x)))", [("return _mean_over_k_of_mean_over_t([np.maximum(0.0, 1.0 - np.asarray(d, dtype=float)) for d in fake_logits])", "return _mean_over_k_of_mean_over_t([np.maximum(0.0, 1.0 + np.asarray(d, dtype=float)) for d in fake_logits])")]),
    ("4-hinge-losses", "feature weight 1 instead of 100", [("lam_feat=100.0", "lam_feat=1.0")]),
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
    skip_torch = True
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
