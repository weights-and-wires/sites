#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/precision-scaling/scripts/check_mutants.py
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
    ("1-effective-parameters", "factor without the minus sign",
     [("np.exp(-p / gamma + offset)", "np.exp(p / gamma + offset)")]),
    ("1-effective-parameters", "offset subtracted instead of added",
     [("np.exp(-p / gamma + offset)", "np.exp(-p / gamma - offset)")]),
    ("1-effective-parameters", "one shared gamma for all three components",
     [('eff_factor(p_a, GAMMA["a"], off["a"])', 'eff_factor(p_a, GAMMA["w"], off["a"])'),
      ('eff_factor(p_kv, GAMMA["kv"], off["kv"])', 'eff_factor(p_kv, GAMMA["w"], off["kv"])')]),
    ("1-effective-parameters", "KV cache factor dropped",
     [('        * eff_factor(p_kv, GAMMA["kv"], off["kv"])\n', "")]),
    ("1-effective-parameters", "offsets always applied",
     [("off = OFFSET if offsets else", "off = OFFSET if True else")]),
    ("1-effective-parameters", "data term uses the wrong sign",
     [("B * d**-ALPHA + E", "B * d**ALPHA + E")]),
    ("1-effective-parameters", "irreducible loss dropped",
     [("B * d**-ALPHA + E", "B * d**-ALPHA")]),
    ("1-effective-parameters", "N_eff not used in the size term",
     [("A * n_eff(n, p_w, p_a, p_kv, offsets) ** -ALPHA", "A * np.asarray(n, dtype=float) ** -ALPHA")]),
    ("2-fit-gamma", "loss gap solved with the wrong sign",
     [("((loss - loss_inf) / A + n**-ALPHA)", "((loss_inf - loss) / A + n**-ALPHA)")]),
    ("2-fit-gamma", "exponent -1/alpha replaced by -alpha",
     [("** (-1.0 / ALPHA)", "** -ALPHA")]),
    ("2-fit-gamma", "fraction not divided by n",
     [("** (-1.0 / ALPHA) / n", "** (-1.0 / ALPHA)")]),
    ("2-fit-gamma", "fit ln f instead of ln(1 - f)",
     [("np.log(1.0 - np.asarray(fracs, dtype=float))", "np.log(np.asarray(fracs, dtype=float))")]),
    ("2-fit-gamma", "returns the rate 1 / gamma",
     [("return float(-np.sum(p * p) / np.sum(p * y))", "return float(-np.sum(p * y) / np.sum(p * p))")]),
    ("2-fit-gamma", "returns a negative gamma",
     [("return float(-np.sum(p * p) / np.sum(p * y))", "return float(np.sum(p * p) / np.sum(p * y))")]),
    ("3-optimal-bits", "cube replaced by a square",
     [("(1.0 - np.exp(-p / gamma_bar)) ** 3", "(1.0 - np.exp(-p / gamma_bar)) ** 2")]),
    ("3-optimal-bits", "cost ignores the bit width",
     [("d = c / (6.0 / 16.0 * n * p)", "d = c / (6.0 * n)")]),
    ("3-optimal-bits", "cost uses the 16-bit factor twice",
     [("d = c / (6.0 / 16.0 * n * p)", "d = c / (6.0 / 16.0 * n * 16)")]),
    ("3-optimal-bits", "root condition exp(x) - 1 = 2x",
     [("- 3.0 * mid < 0", "- 2.0 * mid < 0")]),
    ("3-optimal-bits", "bisection starts at 0 and finds the trivial root",
     [("lo, hi = 1.0, 10.0", "lo, hi = 0.0, 1.0")]),
    ("3-optimal-bits", "P* not scaled by gamma_bar",
     [("return gamma_bar * 0.5 * (lo + hi)", "return 0.5 * (lo + hi)")]),
    ("3-optimal-bits", "returns the highest loss",
     [("if loss.min() < best[1]:", "if loss.min() > best[1] or best[0] is None:")]),
    ("4-ptq-degradation", "data and parameter exponents swapped",
     [("d**GAMMA_D / n**GAMMA_N", "d**GAMMA_N / n**GAMMA_D")]),
    ("4-ptq-degradation", "precision term grows with bits",
     [("np.exp(-p_post / GAMMA_POST)", "np.exp(p_post / GAMMA_POST)")]),
    ("4-ptq-degradation", "no C_T",
     [("return C_T * d**GAMMA_D", "return d**GAMMA_D")]),
    ("4-ptq-degradation", "D_crit exponent 1 / gamma_D",
     [("** (1.0 / (GAMMA_D + BETA))", "** (1.0 / GAMMA_D)")]),
    ("4-ptq-degradation", "D_crit with e^(-p / gamma_post)",
     [("np.exp(p_post / GAMMA_POST)\n    return", "np.exp(-p_post / GAMMA_POST)\n    return")]),
    ("4-ptq-degradation", "D_crit missing the gamma_D factor",
     [("(num / (GAMMA_D * C_T))", "(num / C_T)")]),
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
