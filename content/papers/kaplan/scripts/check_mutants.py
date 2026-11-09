#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/kaplan/scripts/check_mutants.py
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
    ("1-fit-power-law", "fit on log y against x (semilog)",
     [("slope, intercept = np.polyfit(lx, ly, 1)\n    alpha = -slope", "slope, intercept = np.polyfit(np.exp(lx), ly, 1)\n    alpha = -slope")]),
    ("1-fit-power-law", "alpha keeps the sign of the slope",
     [("alpha = -slope", "alpha = slope")]),
    ("1-fit-power-law", "x_c = exp(intercept), no division by alpha",
     [("x_c = np.exp(intercept / alpha)", "x_c = np.exp(intercept)")]),
    ("1-fit-power-law", "log10 for the data but natural log in the scale",
     [("lx, ly = np.log(np.asarray(x, dtype=float)), np.log(np.asarray(y, dtype=float))\n    slope, intercept = np.polyfit(lx, ly, 1)\n    alpha",
       "lx, ly = np.log10(np.asarray(x, dtype=float)), np.log10(np.asarray(y, dtype=float))\n    slope, intercept = np.polyfit(lx, ly, 1)\n    alpha")]),
    ("1-fit-power-law", "uncertainty without the single-measurement term",
     [("s2 * (1 + 1 / n +", "s2 * (1 / n +")]),
    ("1-fit-power-law", "uncertainty ignores the distance from the data",
     [("s2 * (1 + 1 / n + (lnew - lx.mean()) ** 2 / sxx)", "s2 * (1 + 1 / n)")]),
    ("1-fit-power-law", "residual variance divided by n instead of n - 2",
     [("resid @ resid / (n - 2)", "resid @ resid / n")]),
    ("2-params-and-compute", "uses 12 * n_layer * d_model**2 and ignores d_ff",
     [("return 2 * d_model * n_layer * (2 * d_attn + d_ff)", "return 12 * n_layer * d_model**2")]),
    ("2-params-and-compute", "d_attn ignored in N",
     [("(2 * d_attn + d_ff)", "(2 * d_model + d_ff)")]),
    ("2-params-and-compute", "factor 2 instead of 6 (forward pass only)",
     [("return 6 * n_params * batch_tokens * steps / PF_DAY", "return 2 * n_params * batch_tokens * steps / PF_DAY")]),
    ("2-params-and-compute", "PF-day uses 60 minutes by mistake",
     [("PF_DAY = 1e15 * 24 * 3600", "PF_DAY = 1e15 * 24 * 60")]),
    ("2-params-and-compute", "context term uses d_model instead of d_attn",
     [("2 * n_layer * n_ctx * d_attn", "2 * n_layer * n_ctx * d_model")]),
    ("2-params-and-compute", "context term dropped from the forward cost",
     [("return 2 * n + 2 * n_layer * n_ctx * d_attn", "return 2 * n")]),
    ("2-params-and-compute", "embedding term ignores positions",
     [("(n_vocab + n_ctx) * d_model", "n_vocab * d_model")]),
    ("3-loss-n-d", "outer exponent alpha_N instead of alpha_D",
     [("+ d_c / d) ** alpha_d", "+ d_c / d) ** alpha_n")]),
    ("3-loss-n-d", "exponent ratio inverted",
     [("(alpha_n / alpha_d)", "(alpha_d / alpha_n)")]),
    ("3-loss-n-d", "sum of the two single-variable laws",
     [("return ((n_c / n) ** (alpha_n / alpha_d) + d_c / d) ** alpha_d", "return (n_c / n) ** alpha_n + (d_c / d) ** alpha_d")]),
    ("3-loss-n-d", "penalty is a difference, not a ratio",
     [("return loss_nd(n, d) / loss_n_only(n) - 1", "return loss_nd(n, d) - loss_n_only(n)")]),
    ("3-loss-n-d", "data rule uses N ** 1",
     [("n, dtype=float) ** 0.74", "n, dtype=float) ** 1.0")]),
    ("4-compute-optimal", "inverse forgets the minus sign",
     [("loss ** (-1 / ALPHA_CMIN)", "loss ** (1 / ALPHA_CMIN)")]),
    ("4-compute-optimal", "loss law uses the wrong direction",
     [("return (CMIN_C / c) ** ALPHA_CMIN", "return (c / CMIN_C) ** ALPHA_CMIN")]),
    ("4-compute-optimal", "model size exponent 0.5",
     [("N_E, P_N = 1.3e9, 0.73", "N_E, P_N = 1.3e9, 0.5")]),
    ("4-compute-optimal", "token exponent from the overfitting rule (0.54)",
     [("D_E, P_D = 2e10, 0.27", "D_E, P_D = 2e10, 0.54")]),
    ("4-compute-optimal", "critical batch size grows with the loss",
     [("return B_STAR / loss ** (1 / ALPHA_B)", "return B_STAR * loss ** (1 / ALPHA_B)")]),
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
