#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/vae/scripts/check_mutants.py
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
    ("1-gaussian-kl", "sigma instead of sigma^2 in the closed form", [("- np.exp(logvar), axis=-1)", "- np.exp(0.5 * logvar), axis=-1)")]),
    ("1-gaussian-kl", "missing 0.5 factor", [("-0.5 * np.sum(1 + logvar", "-1.0 * np.sum(1 + logvar")]),
    ("1-gaussian-kl", "sign flipped (returns -KL)", [("return -0.5 * np.sum(1 + logvar", "return 0.5 * np.sum(1 + logvar")]),
    ("1-gaussian-kl", "forgets the mu^2 term", [("- mu**2 - np.exp(logvar)", "- np.exp(logvar)")]),
    ("1-gaussian-kl", "sums over the batch axis", [("axis=-1)\n\n\ndef kl_monte_carlo", "axis=0)\n\n\ndef kl_monte_carlo")]),
    ("1-gaussian-kl", "Monte Carlo q density uses logvar as variance in the squared term", [("logvar + eps**2", "logvar + eps**2 * np.exp(logvar)")]),
    ("1-gaussian-kl", "Monte Carlo ignores sigma when forming z", [("z = mu + sigma * eps", "z = mu + eps")]),
    ("2-reparameterization", "exp(logvar) used as the standard deviation", [("np.exp(0.5 * logvar) * eps", "np.exp(logvar) * eps")]),
    ("2-reparameterization", "logvar treated as sigma", [("np.exp(0.5 * logvar) * eps", "logvar * eps")]),
    ("2-reparameterization", "d_sigma forgets eps", [("np.mean(g * eps)", "np.mean(g)")]),
    ("2-reparameterization", "gradient taken at the mean instead of at z", [("g = fprime(z)", "g = fprime(mu + 0 * eps)")]),
    ("3-elbo-bound", "KL added with the wrong sign", [("neg_kl = 0.5 * (1 + logvar - mu**2 - np.exp(logvar))", "neg_kl = -0.5 * (1 + logvar - mu**2 - np.exp(logvar))")]),
    ("3-elbo-bound", "likelihood term uses the mean, not samples", [("log_lik = -0.5 * np.log(2 * np.pi) - 0.5 * (x - z) ** 2", "log_lik = -0.5 * np.log(2 * np.pi) - 0.5 * (x - mu) ** 2")]),
    ("3-elbo-bound", "KL term dropped (reconstruction only)", [("return neg_kl + np.mean(log_lik)", "return np.mean(log_lik)")]),
    ("3-elbo-bound", "sigma computed as exp(logvar)", [("sigma = np.exp(0.5 * logvar)", "sigma = np.exp(logvar)")]),
    ("4-amortized-encoder", "dc ignores the path through the sample", [("d_c = 0.5 * (np.exp(c) - 1) - (x - z) * eps * 0.5 * sigma", "d_c = 0.5 * (np.exp(c) - 1)")]),
    ("4-amortized-encoder", "dw forgets the chain-rule factor x", [("np.mean(d_mu * x)", "np.mean(d_mu)")]),
    ("4-amortized-encoder", "KL gradient for c uses d/dsigma instead of d/dlogvar", [("0.5 * (np.exp(c) - 1) -", "(np.exp(c) - 1) -")]),
    ("4-amortized-encoder", "reconstruction gradient has the wrong sign", [("d_mu = mu - (x - z)", "d_mu = mu + (x - z)")]),
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
