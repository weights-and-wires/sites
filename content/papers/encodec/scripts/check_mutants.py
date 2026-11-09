#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/encodec/scripts/check_mutants.py
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
    ("1-ema-codebook", "np.add.at replaced by fancy-index += (loses repeated indices)", [("np.add.at(sums, idx, batch)", "sums[idx] += batch")]),
    ("1-ema-codebook", "no dead-entry replacement", [("for k in np.flatnonzero(ema_count < threshold):", "for k in []:")]),
    ("1-ema-codebook", "replaced entry keeps its old EMA statistics", [("        ema_sum[k] = v\n        ema_count[k] = 1.0\n", "")]),
    ("1-ema-codebook", "replacement picks the same batch row every time", [("batch[rng.integers(len(batch))]", "batch[0]")]),
    ("1-ema-codebook", "decay and (1 - decay) swapped", [("decay * ema_count + (1 - decay) * counts", "(1 - decay) * ema_count + decay * counts")]),
    ("1-ema-codebook", "entries left as ema_sum (no division by ema_count)", [("codebook = ema_sum / np.maximum(ema_count, eps)[:, None]", "codebook = ema_sum")]),
    ("1-ema-codebook", "perplexity takes exp of an entropy in bits", [("np.exp(-(p * np.log(p)).sum())", "np.exp(-(p * np.log2(p)).sum())")]),
    ("1-ema-codebook", "usage reports the codebook size instead of the entries used", [("return int(len(p)), float", "return int(len(codebook)), float")]),
    ("2-loss-balancer", "weights not divided by their sum", [("frac = weights / weights.sum()", "frac = weights")]),
    ("2-loss-balancer", "divide by the current norm instead of the EMA", [("zip(frac, grads, new_ema)", "zip(frac, grads, norms)")]),
    ("2-loss-balancer", "EMA uses the old value for the division", [("zip(frac, grads, new_ema)", "zip(frac, grads, ema if ema is not None else norms)")]),
    ("2-loss-balancer", "R ignored", [("R * f * g / n", "f * g / n")]),
    ("2-loss-balancer", "weights applied to the raw gradient and the norm", [("R * f * g / n", "R * f * g / (n * f)")]),
    ("2-loss-balancer", "EMA decay swapped", [("beta * np.asarray(ema) + (1 - beta) * norms", "(1 - beta) * np.asarray(ema) + beta * norms")]),
    ("3-multiscale-spectral-loss", "squared L2 term", [("np.linalg.norm(d)", "(d ** 2).sum()")]),
    ("3-multiscale-spectral-loss", "sum over scales instead of the mean", [("return total / len(exponents)", "return total")]),
    ("3-multiscale-spectral-loss", "window 2 ** i replaced by i", [("stft_mag(x, 2**i) - stft_mag(x_hat, 2**i)", "stft_mag(x, 2**(i + 1)) - stft_mag(x_hat, 2**(i + 1))")]),
    ("3-multiscale-spectral-loss", "L1 term dropped", [("np.abs(d).sum() + np.linalg.norm(d)", "np.linalg.norm(d)")]),
    ("3-multiscale-spectral-loss", "L2 term dropped", [("np.abs(d).sum() + np.linalg.norm(d)", "np.abs(d).sum()")]),
    ("3-multiscale-spectral-loss", "time loss is a mean", [("np.abs(np.asarray(x) - np.asarray(x_hat)).sum()", "np.abs(np.asarray(x) - np.asarray(x_hat)).mean()")]),
    ("4-entropy-bitrate", "strides summed instead of multiplied", [("fs / math.prod(strides)", "fs / sum(strides)")]),
    ("4-entropy-bitrate", "natural log instead of log2 in the entropy", [("-(p * np.log2(p)).sum()", "-(p * np.log(p)).sum()")]),
    ("4-entropy-bitrate", "zero probabilities not filtered", [("        p = p[p > 0]\n", "")]),
    ("4-entropy-bitrate", "entropy averaged over codebooks instead of summed", [("return float(bits * _steps(fs, strides) / 1000)", "return float(bits / len(probs) * _steps(fs, strides) / 1000)")]),
    ("4-entropy-bitrate", "bits per code fixed at 10 regardless of codebook size", [("math.log2(n_codes)", "10")]),
    ("4-entropy-bitrate", "n_q rounded silently instead of raising", [("    if r < 1 or abs(n - r) > 1e-6:\n", "    if False:\n")]),
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
