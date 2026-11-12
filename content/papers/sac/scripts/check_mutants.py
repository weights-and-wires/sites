#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/sac/scripts/check_mutants.py
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
    ("1-soft-value", "policy: no max subtraction (overflows)", [("e = np.exp(z - np.max(z))", "e = np.exp(z)")]),
    ("1-soft-value", "policy: temperature ignored", [("z = np.asarray(q, dtype=float) / alpha\n    e", "z = np.asarray(q, dtype=float)\n    e")]),
    ("1-soft-value", "policy: temperature multiplies instead of divides", [("z = np.asarray(q, dtype=float) / alpha\n    e", "z = np.asarray(q, dtype=float) * alpha\n    e")]),
    ("1-soft-value", "value: no max subtraction (overflows)", [("m = np.max(z)\n    return float(alpha * (m + np.log(np.sum(np.exp(z - m)))))", "return float(alpha * np.log(np.sum(np.exp(z))))")]),
    ("1-soft-value", "value: forgets to add the shift back", [("alpha * (m + np.log(", "alpha * (np.log(")]),
    ("1-soft-value", "value: no outer alpha", [("float(alpha * (m +", "float((m +")]),
    ("1-soft-value", "value: hard max instead of logsumexp", [("return float(alpha * (m + np.log(np.sum(np.exp(z - m)))))", "return float(alpha * m)")]),
    ("1-soft-value", "value: mean instead of sum inside the log", [("np.sum(np.exp(z - m))", "np.mean(np.exp(z - m))")]),
    ("2-squashed-log-prob", "correction added instead of subtracted", [("log_normal - log_one_minus_tanh_sq", "log_normal + log_one_minus_tanh_sq")]),
    ("2-squashed-log-prob", "no correction", [("log_normal - log_one_minus_tanh_sq", "log_normal")]),
    ("2-squashed-log-prob", "no sum over dimensions", [("np.sum(log_normal - log_one_minus_tanh_sq, axis=-1)", "(log_normal - log_one_minus_tanh_sq)")]),
    ("2-squashed-log-prob", "sum over every axis", [("axis=-1)", "axis=None)")]),
    ("2-squashed-log-prob", "no log std term", [(" - np.log(std) - 0.5", " - 0.5")]),
    ("2-squashed-log-prob", "naive 1 - tanh^2 (infinite at u = 25)", [("2.0 * (np.log(2.0) - u - np.logaddexp(0.0, -2.0 * u))", "np.log(1.0 - np.tanh(u) ** 2)")]),
    ("2-squashed-log-prob", "tanh instead of tanh squared", [("2.0 * (np.log(2.0) - u - np.logaddexp(0.0, -2.0 * u))", "np.log(1.0 - np.tanh(u))")]),
    ("2-squashed-log-prob", "variance instead of std in the Gaussian", [("((u - mu) / std) ** 2", "((u - mu) / std**2) ** 2")]),
    ("3-soft-targets", "value target: maximum of the critics", [("np.minimum(q1, q2) - logp", "np.maximum(q1, q2) - logp")]),
    ("3-soft-targets", "value target: mean of the critics", [("np.minimum(q1, q2) - logp", "0.5 * (q1 + q2) - logp")]),
    ("3-soft-targets", "value target: first critic only", [("np.minimum(q1, q2) - logp", "q1 - logp")]),
    ("3-soft-targets", "value target: entropy sign flipped", [("np.minimum(q1, q2) - logp", "np.minimum(q1, q2) + logp")]),
    ("3-soft-targets", "Q target: no terminal mask", [("gamma * (1.0 - done) * v_next", "gamma * v_next")]),
    ("3-soft-targets", "Q target: gamma on the reward", [("return r + gamma * (1.0 - done) * v_next", "return gamma * (r + (1.0 - done) * v_next)")]),
    ("3-soft-targets", "Q target: mask inverted", [("(1.0 - done)", "done")]),
    ("3-soft-targets", "loss: no one half", [("0.5 * (pred - target) ** 2", "(pred - target) ** 2")]),
    ("3-soft-targets", "loss: absolute error", [("0.5 * (pred - target) ** 2", "0.5 * np.abs(pred - target)")]),
    ("3-soft-targets", "loss: sum instead of mean", [("np.mean(0.5 * (pred - target) ** 2)", "np.sum(0.5 * (pred - target) ** 2)")]),
    ("4-reparam-gradient", "reparam: d_sigma forgets eps", [("float(np.mean(g * eps))", "float(np.mean(g))")]),
    ("4-reparam-gradient", "reparam: uses Q instead of Q'", [("g = dq(mu + sigma * eps)", "g = -((mu + sigma * eps - 2.0) ** 2)")]),
    ("4-reparam-gradient", "reparam: d_mu scaled by sigma", [("float(np.mean(g)), float(np.mean(g * eps))", "float(np.mean(g) * sigma), float(np.mean(g * eps))")]),
    ("4-reparam-gradient", "score: score divided by sigma, not sigma squared", [("(a - mu) / sigma**2", "(a - mu) / sigma")]),
    ("4-reparam-gradient", "score: sigma score drops the -1/sigma term", [("(a - mu) ** 2 / sigma**3 - 1.0 / sigma", "(a - mu) ** 2 / sigma**3")]),
    ("4-reparam-gradient", "score: weights by Q' instead of Q", [("value = q(a)", "value = -2.0 * (a - 2.0)")]),
    ("4-reparam-gradient", "score: score of the wrong sign", [("score_mu = (a - mu) / sigma**2", "score_mu = (mu - a) / sigma**2")]),
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
        folder, name, edits = entry
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
