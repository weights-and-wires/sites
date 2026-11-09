#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/gan/scripts/check_mutants.py
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
    ("1-optimal-discriminator", "D* with p_g in the numerator (roles swapped)", [("np.divide(p_data, denom,", "np.divide(p_g, denom,")]),
    ("1-optimal-discriminator", "D* without the 0.5 fill (nan where both are 0)", [("out = np.full_like(denom, 0.5)", "out = np.full_like(denom, np.nan)")]),
    ("1-optimal-discriminator", "value without the 0 log 0 mask", [
        ("t1 = np.where(p_data > 0, p_data * np.log(np.where(p_data > 0, D, 1.0)), 0.0)", "t1 = p_data * np.log(D)"),
        ("t2 = np.where(p_g > 0, p_g * np.log(np.where(p_g > 0, 1.0 - D, 1.0)), 0.0)", "t2 = p_g * np.log(1.0 - D)")]),
    ("1-optimal-discriminator", "value uses log D for the fake term", [("np.log(np.where(p_g > 0, 1.0 - D, 1.0))", "np.log(np.where(p_g > 0, D, 1.0))")]),
    ("1-optimal-discriminator", "JSD without the two 1/2 factors", [("return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)", "return _kl(p, m) + _kl(q, m)")]),
    ("1-optimal-discriminator", "JSD compares against q instead of the mixture", [("return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)", "return 0.5 * _kl(p, q) + 0.5 * _kl(q, p)")]),
    ("2-generator-losses", "log_sigmoid via log(sigmoid(a)) (overflows)", [("return -np.logaddexp(0.0, -np.asarray(a, dtype=float))", "a = np.asarray(a, dtype=float)\n    return np.log(1 / (1 + np.exp(-a)))")]),
    ("2-generator-losses", "discriminator loss uses log D(G(z)) for fakes", [("np.mean(log_sigmoid(-np.asarray(fake_logits, dtype=float)))", "np.mean(log_sigmoid(np.asarray(fake_logits, dtype=float)))")]),
    ("2-generator-losses", "non-saturating loss has the wrong sign", [("return float(-np.mean(log_sigmoid(a)))", "return float(np.mean(log_sigmoid(a)))")]),
    ("2-generator-losses", "saturating gradient equals the non-saturating one", [("return -np.exp(log_sigmoid(a)) / m", "return -np.exp(log_sigmoid(-a)) / m")]),
    ("2-generator-losses", "gradient omits the 1/m factor", [("return -np.exp(log_sigmoid(-a)) / m", "return -np.exp(log_sigmoid(-a))")]),
    ("3-train-1d-gan", "discriminator gradient with the fake sign flipped", [("+ features(x_fake).T @ d_fake / m", "- features(x_fake).T @ d_fake / m")]),
    ("3-train-1d-gan", "generator gradient for the saturating loss", [("dx = -(1 - d) * (pd[0]", "dx = -d * (pd[0]")]),
    ("3-train-1d-gan", "chain rule drops exp(s) in d x / d s", [("(dx * np.exp(s) * z).sum()", "(dx * z).sum()")]),
    ("3-train-1d-gan", "chain rule drops the x^2 feature derivative", [("(pd[0] + 2 * pd[1] * x)", "pd[0]")]),
    ("4-mode-collapse", "histogram normalizes by len(samples)", [("return counts / counts.sum()", "return counts / len(samples)")]),
    ("4-mode-collapse", "JS divergence with a log base 2", [("return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)", "return (0.5 * _kl(p, m) + 0.5 * _kl(q, m)) / np.log(2)")]),
    ("4-mode-collapse", "mode counts ignore min_frac", [(">= min_frac", "> 0")]),
    ("4-mode-collapse", "radius test is inclusive", [("< radius", "<= radius")]),
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
