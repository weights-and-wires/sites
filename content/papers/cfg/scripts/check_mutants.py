#!/usr/bin/env python3
"""Check that the cfg exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/cfg/scripts/check_mutants.py
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
    ("1-logsnr-and-dropout", "alpha^2 = sigmoid(-lambda): signal and noise swapped", [("alpha_sq = 1.0 / (1.0 + np.exp(-lam))", "alpha_sq = 1.0 / (1.0 + np.exp(lam))")]),
    ("1-logsnr-and-dropout", "sigma = 1 - alpha instead of sqrt(1 - alpha^2)", [("np.sqrt(1.0 - alpha_sq)", "1.0 - np.sqrt(alpha_sq)")]),
    ("1-logsnr-and-dropout", "lambda mapping runs upward (u = 0 gives lambda_min)", [("return -2.0 * np.log(np.tan(a * np.asarray(u, dtype=float) + b))", "return 2.0 * np.log(np.tan(a * np.asarray(u, dtype=float) + b))")]),
    ("1-logsnr-and-dropout", "lambda drawn uniformly instead of through the tan map", [("return -2.0 * np.log(np.tan(a * np.asarray(u, dtype=float) + b))", "return lam_max + (lam_min - lam_max) * np.asarray(u, dtype=float)")]),
    ("1-logsnr-and-dropout", "one coin flip for the whole batch", [("drop = rng.random(c.shape) < p_uncond", "drop = np.full(c.shape, rng.random() < p_uncond)")]),
    ("1-logsnr-and-dropout", "drop probability inverted (keeps with p_uncond)", [("drop = rng.random(c.shape) < p_uncond", "drop = rng.random(c.shape) >= p_uncond")]),
    ("1-logsnr-and-dropout", "drop_condition edits the caller's array", [("return np.where(drop, null, c)", "c[drop] = null\n    return c")]),
    ("1-logsnr-and-dropout", "corrupt ignores per-row lambda (uses the first)", [("alpha, sigma = alpha_sigma(lam)\n    shape", "alpha, sigma = alpha_sigma(np.ravel(lam)[0])\n    shape")]),
    ("2-guided-score", "paper's w read as the scale s = 1 + w", [("return (1.0 + w) * np.asarray(eps_c) - w * np.asarray(eps_u)", "return w * np.asarray(eps_c) + (1.0 - w) * np.asarray(eps_u)")]),
    ("2-guided-score", "w * eps_c - (w - 1) * eps_u with w taken as the scale", [("return (1.0 + w) * np.asarray(eps_c) - w * np.asarray(eps_u)", "return np.asarray(eps_u) + w * (np.asarray(eps_c) - np.asarray(eps_u))")]),
    ("2-guided-score", "guidance applied to the unconditional prediction", [("return (1.0 + w) * np.asarray(eps_c) - w * np.asarray(eps_u)", "return (1.0 + w) * np.asarray(eps_u) - w * np.asarray(eps_c)")]),
    ("2-guided-score", "implicit gradient without the 1 / sigma factor", [("return -(np.asarray(eps_c) - np.asarray(eps_u)) / sigma", "return -(np.asarray(eps_c) - np.asarray(eps_u))")]),
    ("2-guided-score", "implicit gradient with the wrong sign", [("return -(np.asarray(eps_c) - np.asarray(eps_u)) / sigma", "return (np.asarray(eps_c) - np.asarray(eps_u)) / sigma")]),
    ("2-guided-score", "classifier guidance adds the gradient", [("return np.asarray(eps_c) - w * sigma * np.asarray(grad_log_pc)", "return np.asarray(eps_c) + w * sigma * np.asarray(grad_log_pc)")]),
    ("2-guided-score", "classifier guidance without the sigma factor", [("return np.asarray(eps_c) - w * sigma * np.asarray(grad_log_pc)", "return np.asarray(eps_c) - w * np.asarray(grad_log_pc)")]),
    ("3-guided-sampler", "guidance weight w used as the scale (1 + w dropped)", [("eps = (1.0 + w) * eps_model(z, c, lam[t]) - w * eps_model(z, NULL, lam[t])", "eps = w * eps_model(z, c, lam[t]) - (w - 1.0) * eps_model(z, NULL, lam[t])")]),
    ("3-guided-sampler", "the unconditional branch is also given the class label", [("w * eps_model(z, NULL, lam[t])", "w * eps_model(z, c, lam[t])")]),
    ("3-guided-sampler", "guidance ignored (conditional prediction only)", [("eps = (1.0 + w) * eps_model(z, c, lam[t]) - w * eps_model(z, NULL, lam[t])", "eps = eps_model(z, c, lam[t])")]),
    ("3-guided-sampler", "an extra noise draw at the last step", [("        if t == T - 1:\n            return x\n", "        if t == T - 1:\n            rng.standard_normal(n)\n            return x\n")]),
    ("3-guided-sampler", "returns z_T instead of the x_T estimate", [("        if t == T - 1:\n            return x\n", "        if t == T - 1:\n            return z\n")]),
    ("3-guided-sampler", "v ignored (always the posterior variance)", [("var = var_tilde ** (1.0 - v) * var_fwd**v", "var = var_tilde")]),
    ("3-guided-sampler", "mean uses z in place of x_t (no denoised estimate)", [("(1.0 - r) * alpha2 * x", "(1.0 - r) * alpha2 * z")]),
    ("3-guided-sampler", "sigma used as variance (sqrt dropped)", [("np.sqrt(var) * rng.standard_normal(n)", "var * rng.standard_normal(n)")]),
    ("4-guided-density", "(1 + w) read as w: p(x|c)^w p(x)^(-w)", [("logits = (1.0 + w) * np.asarray(log_pc) - w * np.asarray(log_p)", "logits = w * np.asarray(log_pc) - w * np.asarray(log_p)")]),
    ("4-guided-density", "sign of the marginal term flipped", [("- w * np.asarray(log_p)", "+ w * np.asarray(log_p)")]),
    ("4-guided-density", "no max subtraction (overflows on large logs)", [("logits = logits - logits.max()", "pass")]),
    ("4-guided-density", "normalized without the grid spacing", [("return p / (p.sum() * dx)", "return p / p.sum()")]),
    ("4-guided-density", "variance around zero instead of the mean", [("var = np.sum((x - mean) ** 2 * p) * dx", "var = np.sum(x**2 * p) * dx")]),
    ("4-guided-density", "std reported as variance", [("return mean, np.sqrt(var)", "return mean, var")]),
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
