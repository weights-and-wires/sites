#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/ldm/scripts/check_mutants.py 
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
    ("1-compute-savings", "value ratio ignores the 3 RGB channels", [("return (H * W * 3) / (h * w * c)", "return (H * W) / (h * w * c)")]),
    ("1-compute-savings", "attention ratio is f squared", [("return (H * W) ** 2 / (h * w) ** 2", "return (H * W) / (h * w)")]),
    ("1-compute-savings", "h and w swapped", [("return H // f, W // f, c", "return W // f, H // f, c")]),
    ("1-compute-savings", "no divisibility check", [("if H % f != 0 or W % f != 0:", "if False:")]),
    ("2-kl-latent-rescale", "sigma taken as exp(logvar)", [("np.exp(0.5 * logvar) * eps", "np.exp(logvar) * eps")]),
    ("2-kl-latent-rescale", "KL drops the -log sigma^2 term", [("- 1.0 - logvar)", "- 1.0)")]),
    ("2-kl-latent-rescale", "KL sums over the batch too", [("kl.sum(axis=(1, 2, 3))", "kl.sum(axis=(0, 1, 2, 3)) * np.ones(kl.shape[0])")]),
    ("2-kl-latent-rescale", "variance instead of std", [("np.sqrt(np.mean((z - z.mean()) ** 2))", "np.mean((z - z.mean()) ** 2)")]),
    ("2-kl-latent-rescale", "per-channel mean inside the std", [("np.sqrt(np.mean((z - z.mean()) ** 2))", "np.sqrt(np.mean((z - z.mean(axis=(0, 2, 3), keepdims=True)) ** 2))")]),
    ("2-kl-latent-rescale", "rescale multiplies", [("return z / sigma_hat", "return z * sigma_hat")]),
    ("3-cross-attention", "softmax over the queries (axis 0)", [("weights = softmax(Q @ K.T / np.sqrt(d), axis=-1)", "weights = softmax(Q @ K.T / np.sqrt(d), axis=0)")]),
    ("3-cross-attention", "keys built from the U-Net features", [("K = tau @ W_K.T", "K = phi @ W_Q.T")]),
    ("3-cross-attention", "values built from the U-Net features", [("V = tau @ W_V.T", "V = phi @ W_Q.T")]),
    ("3-cross-attention", "scale by sqrt(d_tau)", [("d = Q.shape[-1]", "d = tau.shape[-1]")]),
    ("3-cross-attention", "no scaling", [("/ np.sqrt(d)", "")]),
    ("4-latent-diffusion-pca", "PCA without centering", [("_, _, Vt = np.linalg.svd(X - mean, full_matrices=False)", "_, _, Vt = np.linalg.svd(X, full_matrices=False)")]),
    ("4-latent-diffusion-pca", "smallest components kept", [("return mean, Vt[:k]", "return mean, Vt[-k:]")]),
    ("4-latent-diffusion-pca", "decode forgets the mean", [("return Z @ W + mean", "return Z @ W")]),
    ("4-latent-diffusion-pca", "encode forgets the mean", [("return (X - mean) @ W.T", "return X @ W.T")]),
    ("4-latent-diffusion-pca", "eps ignores the latent mean", [("(zt - np.sqrt(abar) * m).T", "zt.T")]),
    ("4-latent-diffusion-pca", "Sigma_t = Sigma + I", [("abar * Sigma + (1.0 - abar) * np.eye(k)", "Sigma + np.eye(k)")]),
    ("4-latent-diffusion-pca", "missing sqrt(1 - abar) factor", [("return np.sqrt(1.0 - abar) * np.linalg.solve", "return np.linalg.solve")]),
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
        folder, name, edits = entry[:3]
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
