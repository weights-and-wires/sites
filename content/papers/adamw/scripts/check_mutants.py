#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/adamw/scripts/check_mutants.py
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
    ("1-adam-l2-vs-adamw", "AdamW decay scaled by alpha", [("+ lam * theta)", "+ alpha * lam * theta)")]),
    ("1-adam-l2-vs-adamw", "AdamW decay inside the adaptive fraction", [("theta - (alpha * m_hat / (np.sqrt(v_hat) + eps) + lam * theta)", "theta - alpha * (m_hat + lam * theta) / (np.sqrt(v_hat) + eps)")]),
    ("1-adam-l2-vs-adamw", "AdamW feeds lam * theta into the moments", [("m_new = beta1 * m + (1 - beta1) * grad\n    v_new = beta2 * v + (1 - beta2) * grad * grad", "g = grad + lam * theta\n    m_new = beta1 * m + (1 - beta1) * g\n    v_new = beta2 * v + (1 - beta2) * g * g")]),
    ("1-adam-l2-vs-adamw", "AdamW step dropped the decay", [(" + lam * theta)", ")")]),
    ("1-adam-l2-vs-adamw", "Adam+L2 applies the decay outside the moments", [("g = grad + lam * theta\n    m_new", "g = grad\n    m_new"), ("alpha * m_hat / (np.sqrt(v_hat) + eps)\n    return theta_new, m_new, v_new\n\n\ndef adamw", "alpha * m_hat / (np.sqrt(v_hat) + eps) + lam * theta\n    return theta_new, m_new, v_new\n\n\ndef adamw")]),
    ("1-adam-l2-vs-adamw", "Adam+L2 forgets the bias correction on v", [("v_hat = v_new / (1 - beta2**t)\n    theta_new = theta - alpha * m_hat", "v_hat = v_new\n    theta_new = theta - alpha * m_hat")]),
    ("2-sgdw-equivalence", "lambda' = lambda (no division by alpha)", [("return lam / alpha", "return lam")]),
    ("2-sgdw-equivalence", "lambda' = lambda * alpha", [("return lam / alpha", "return lam * alpha")]),
    ("2-sgdw-equivalence", "SGDW decay goes through the momentum", [("m_new = beta1 * m + alpha * grad\n    return theta - m_new - lam * theta, m_new", "m_new = beta1 * m + alpha * (grad + lam * theta)\n    return theta - m_new, m_new")]),
    ("2-sgdw-equivalence", "SGDW decay scaled by alpha", [("- lam * theta, m_new", "- alpha * lam * theta, m_new")]),
    ("2-sgdw-equivalence", "L2 gradient term added after the momentum", [("g = grad + lam * theta\n    m_new = beta1 * m + alpha * g\n    return theta - m_new, m_new", "m_new = beta1 * m + alpha * grad\n    return theta - m_new - alpha * lam * theta, m_new")]),
    ("3-fixed-preconditioner", "equivalent_l2 forgets the scale s", [("return lam / alpha * np.asarray(s, dtype=float)", "return lam / alpha * np.ones_like(np.asarray(s, dtype=float))")]),
    ("3-fixed-preconditioner", "equivalent_l2 divides by s", [("lam / alpha * np.asarray(s, dtype=float)", "lam / alpha / np.asarray(s, dtype=float)")]),
    ("3-fixed-preconditioner", "L2 term not divided by s", [("return theta - alpha * (grad + lam_l2 * theta) / s", "return theta - alpha * grad / s - alpha * lam_l2 * theta")]),
    ("3-fixed-preconditioner", "decoupled decay divided by s", [("return (1 - lam) * theta - alpha * grad / s", "return theta - lam * theta / s - alpha * grad / s")]),
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
    skip_torch = False
    survivors = 0
    ran = 0
    for entry in MUTANTS:
        folder, name, edits = entry[:3]
        if len(entry) == 4 and skip_torch:
            print(f"SKIP  {folder}: {name} (torch not available)")
            continue
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
