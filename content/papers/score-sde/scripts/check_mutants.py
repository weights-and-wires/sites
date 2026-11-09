#!/usr/bin/env python3
"""Check that the score-SDE exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/score-sde/scripts/check_mutants.py
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
    ("1-vp-marginal", "int_beta drops the 1/2 on the quadratic term", [("bmin * t + 0.5 * (bmax - bmin) * t**2", "bmin * t + (bmax - bmin) * t**2")]),
    ("1-vp-marginal", "mean scale without the 1/2 in the exponent", [("np.exp(-0.5 * ib), 1.0 - np.exp(-ib)", "np.exp(-ib), 1.0 - np.exp(-ib)")]),
    ("1-vp-marginal", "variance is exp(-int beta) instead of 1 - exp(-int beta)", [("np.exp(-0.5 * ib), 1.0 - np.exp(-ib)", "np.exp(-0.5 * ib), np.exp(-ib)")]),
    ("1-vp-marginal", "Euler-Maruyama drift without the 1/2", [("x = x - 0.5 * b * x * dt +", "x = x - b * x * dt +")]),
    ("1-vp-marginal", "noise scaled by dt instead of sqrt(dt)", [("np.sqrt(b * dt) * rng", "np.sqrt(b) * dt * rng")]),
    ("1-vp-marginal", "diffusion g = beta instead of sqrt(beta)", [("np.sqrt(b * dt) * rng", "b * np.sqrt(dt) * rng")]),
    ("1-vp-marginal", "no noise term (deterministic decay)", [("+ np.sqrt(b * dt) * rng.standard_normal(x.shape)", "")]),
    ("2-reverse-and-flow-drift", "ODE drift uses the full g^2 on the score", [("return -0.5 * beta(t) * x - 0.5 * beta(t) * score", "return -0.5 * beta(t) * x - beta(t) * score")]),
    ("2-reverse-and-flow-drift", "SDE drift uses g^2 / 2 on the score", [("return -0.5 * beta(t) * x - beta(t) * score", "return -0.5 * beta(t) * x - 0.5 * beta(t) * score")]),
    ("2-reverse-and-flow-drift", "sign of the score term flipped", [("return -0.5 * beta(t) * x - 0.5 * beta(t) * score", "return -0.5 * beta(t) * x + 0.5 * beta(t) * score")]),
    ("2-reverse-and-flow-drift", "score ignores the data variance (uses sigma^2 only)", [("var = m**2 * s0**2 + 1.0 - m**2", "var = 1.0 - m**2")]),
    ("2-reverse-and-flow-drift", "score centered at m0 instead of m(t) m0", [("-(x - m * m0) / var", "-(x - m0) / var")]),
    ("3-score-matching-loss", "no lambda weighting", [("np.mean(var[:, 0] * np.sum(err**2, axis=-1))", "np.mean(np.sum(err**2, axis=-1))")]),
    ("3-score-matching-loss", "lambda = 1 / sigma^2 instead of sigma^2", [("np.mean(var[:, 0] * np.sum(err**2, axis=-1))", "np.mean(np.sum(err**2, axis=-1) / var[:, 0])")]),
    ("3-score-matching-loss", "model sees x0 instead of x(t)", [("err = score_model(x_t, t) - target", "err = score_model(x0, t) - target")]),
    ("3-score-matching-loss", "target score uses sigma instead of sigma^2 in the denominator", [("target = -(x_t - m * x0) / var", "target = -(x_t - m * x0) / np.sqrt(var)")]),
    ("3-score-matching-loss", "noise scaled by sigma^2 instead of sigma", [("x_t = m * x0 + np.sqrt(var) * eps", "x_t = m * x0 + var * eps")]),
    ("3-score-matching-loss", "loss averages over data dimensions", [("np.sum(err**2, axis=-1)", "np.mean(err**2, axis=-1)")]),
    ("4-flow-ode-sampler", "integrates forward in time (eps to 1)", [("np.linspace(1.0, eps, n_steps + 1)", "np.linspace(eps, 1.0, n_steps + 1)")]),
    ("4-flow-ode-sampler", "full score coefficient (SDE drift without noise)", [("- 0.5 * beta(t) * score_fn(x, t)", "- beta(t) * score_fn(x, t)")]),
    ("4-flow-ode-sampler", "time at the end of the step", [("t = ts[k]\n", "t = ts[k + 1]\n")]),
    ("4-flow-ode-sampler", "positive dt (wrong sign)", [("dt = ts[k + 1] - ts[k]", "dt = ts[k] - ts[k + 1]")]),
    ("4-flow-ode-sampler", "forward drift only (score ignored)", [("- 0.5 * beta(t) * score_fn(x, t)", "")]),
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
