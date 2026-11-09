#!/usr/bin/env python3
"""Check that the DDPM exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/ddpm/scripts/check_mutants.py
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

SQ = "return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps"

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-forward-process", "alpha_bar is the per-step alpha, not the cumulative product", [("alpha_bars = np.cumprod(alphas)", "alpha_bars = alphas")]),
    ("1-forward-process", "off-by-one: t indexes alpha_bars directly", [("alpha_bars[np.asarray(t) - 1]", "alpha_bars[np.asarray(t)]")]),
    ("1-forward-process", "alpha_bar used without the square root on the signal", [(SQ, "return ab * x0 + np.sqrt(1.0 - ab) * eps")]),
    ("1-forward-process", "noise scale 1 - alpha_bar instead of its square root", [(SQ, "return np.sqrt(ab) * x0 + (1.0 - ab) * eps")]),
    ("1-forward-process", "signal and noise scales both sqrt(alpha_bar)", [(SQ, "return np.sqrt(ab) * x0 + np.sqrt(ab) * eps")]),
    ("1-forward-process", "one alpha_bar for the whole batch (ignores per-row t)", [("ab = alpha_bars[np.asarray(t) - 1]", "ab = alpha_bars[np.asarray(t).ravel()[0] - 1]")]),
    ("2-posterior-and-mean", "variance beta_t instead of beta_tilde_t", [("var = (1 - ab_prev) / (1 - ab_t) * beta_t", "var = beta_t")]),
    ("2-posterior-and-mean", "alpha_bar_{t-1} read as alpha_bars[t - 2] at t = 1 (wraps to alpha_bar_T)", [("ab_prev = 1.0 if t == 1 else alpha_bars[t - 2]", "ab_prev = alpha_bars[t - 2]")]),
    ("2-posterior-and-mean", "coefficients on x0 and x_t swapped", [
        ("mean = (np.sqrt(ab_prev) * beta_t / (1 - ab_t)) * x0 + (np.sqrt(alpha_t) * (1 - ab_prev) / (1 - ab_t)) * x_t",
         "mean = (np.sqrt(ab_prev) * beta_t / (1 - ab_t)) * x_t + (np.sqrt(alpha_t) * (1 - ab_prev) / (1 - ab_t)) * x0")]),
    ("2-posterior-and-mean", "Equation 11 without the 1 / sqrt(alpha_t) factor", [("return (x_t - beta_t / np.sqrt(1 - ab_t) * eps_pred) / np.sqrt(alpha_t)", "return x_t - beta_t / np.sqrt(1 - ab_t) * eps_pred")]),
    ("2-posterior-and-mean", "Equation 11 divides by (1 - alpha_bar) instead of its square root", [("beta_t / np.sqrt(1 - ab_t) * eps_pred", "beta_t / (1 - ab_t) * eps_pred")]),
    ("3-loss-and-sampler", "model sees x0 instead of x_t", [("err = eps - eps_model(x_t, t)", "err = eps - eps_model(x0, t)")]),
    ("3-loss-and-sampler", "loss averages over the data dimensions too", [("np.mean(np.sum(err**2, axis=-1))", "np.mean(err**2)")]),
    ("3-loss-and-sampler", "regress on x0 instead of epsilon", [("err = eps - eps_model(x_t, t)", "err = x0 - eps_model(x_t, t)")]),
    ("3-loss-and-sampler", "sampler adds noise at the last step", [("z = rng.standard_normal(shape) if t > 1 else 0.0", "z = rng.standard_normal(shape)")]),
    ("3-loss-and-sampler", "sampler never adds noise", [("z = rng.standard_normal(shape) if t > 1 else 0.0", "z = 0.0")]),
    ("3-loss-and-sampler", "divides by alpha_t instead of sqrt(alpha_t)", [("/ np.sqrt(alphas[t - 1]) + np.sqrt(var) * z", "/ alphas[t - 1] + np.sqrt(var) * z")]),
    ("3-loss-and-sampler", "sigma option ignored (always beta_tilde)", [('if sigma == "beta":', 'if False:')]),
    ("3-loss-and-sampler", "noise coefficient sigma^2 instead of sigma", [("np.sqrt(var) * z", "var * z")]),
    ("3-loss-and-sampler", "eps coefficient beta_t / sqrt(1 - alpha_bar) replaced by (1 - alpha_bar)", [("(1 - alphas[t - 1]) / np.sqrt(1 - alpha_bars[t - 1])", "(1 - alpha_bars[t - 1])")]),
    ("4-bound-weights", "beta_t not squared in the weight", [("beta_t**2 / (2 * sigma2", "beta_t / (2 * sigma2")]),
    ("4-bound-weights", "missing the alpha_t factor", [("(2 * sigma2 * alpha_t * (1 - alpha_bars[t - 1]))", "(2 * sigma2 * (1 - alpha_bars[t - 1]))")]),
    ("4-bound-weights", "1 - alpha_bar replaced by alpha_bar", [("alpha_t * (1 - alpha_bars[t - 1]))", "alpha_t * alpha_bars[t - 1])")]),
    ("4-bound-weights", "factor 2 forgotten", [("(2 * sigma2 * alpha_t", "(sigma2 * alpha_t")]),
    ("4-bound-weights", "mean over dimensions instead of sum", [("np.sum((np.asarray(eps) - np.asarray(eps_pred)) ** 2)", "np.mean((np.asarray(eps) - np.asarray(eps_pred)) ** 2)")]),
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
