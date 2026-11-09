#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/trpo/scripts/check_mutants.py
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
    ("1-surrogate-objective", "visitation uses the matrix, not its transpose", [("np.linalg.solve((np.eye(len(rho0)) - gamma * P_pi).T, rho0)", "np.linalg.solve(np.eye(len(rho0)) - gamma * P_pi, rho0)")]),
    ("1-surrogate-objective", "visitation normalized to sum to 1", [("rho0)\n\n\ndef advantages", "rho0) * (1 - gamma)\n\n\ndef advantages")]),
    ("1-surrogate-objective", "Q without the discount on V(s')", [("r[:, None] + gamma * np.einsum", "r[:, None] + np.einsum")]),
    ("1-surrogate-objective", "advantage is Q minus the mean of Q over actions, uniformly", [("return Q - V[:, None]", "return Q - Q.mean(axis=1, keepdims=True)")]),
    ("1-surrogate-objective", "surrogate drops the eta(pi) constant", [("return eta + np.sum(rho[:, None] * pi_new * A)", "return np.sum(rho[:, None] * pi_new * A)")]),
    ("1-surrogate-objective", "surrogate weights advantages by the old policy", [("rho[:, None] * pi_new * A", "rho[:, None] * pi * A")]),
    ("2-improvement-bound", "KL arguments swapped", [("np.sum(pi * np.log(pi / pi_new), axis=1).max()", "np.sum(pi_new * np.log(pi_new / pi), axis=1).max()")]),
    ("2-improvement-bound", "total variation without the 1/2", [("return 0.5 * np.abs(pi - pi_new).sum(axis=1).max()", "return np.abs(pi - pi_new).sum(axis=1).max()")]),
    ("2-improvement-bound", "mean KL over states instead of the max", [("np.sum(pi * np.log(pi / pi_new), axis=1).max()", "np.sum(pi * np.log(pi / pi_new), axis=1).mean()")]),
    ("2-improvement-bound", "total variation averaged over states", [("0.5 * np.abs(pi - pi_new).sum(axis=1).max()", "0.5 * np.abs(pi - pi_new).sum(axis=1).mean()")]),
    ("2-improvement-bound", "(1 - gamma) not squared in C", [("(1 - gamma) ** 2", "(1 - gamma)")]),
    ("2-improvement-bound", "TV bound forgets to square the divergence", [("max_tv(pi, pi_new) ** 2", "max_tv(pi, pi_new)")]),
    ("2-improvement-bound", "C uses the factor 2 from Equation 6 instead of 4", [("return 4 * epsilon", "return 2 * epsilon")]),
    ("3-conjugate-gradient", "steepest descent: new direction is just the residual", [("p = r + (rs_new / rs) * p", "p = r")]),
    ("3-conjugate-gradient", "step size uses p.p instead of p.Ap", [("alpha = rs / (p @ Ap)", "alpha = rs / (p @ p)")]),
    ("3-conjugate-gradient", "residual not updated after each step", [("r = r - alpha * Ap", "pass")]),
    ("3-conjugate-gradient", "no early stop on a small residual", [("if rs < tol:\n            break\n        Ap", "Ap")]),
    ("3-conjugate-gradient", "forgets the 2 in the step length", [("np.sqrt(2 * delta / (s @ As))", "np.sqrt(delta / (s @ As))")]),
    ("3-conjugate-gradient", "step length without the square root", [("np.sqrt(2 * delta / (s @ As))", "2 * delta / (s @ As)")]),
    ("4-trpo-step", "Fisher-vector product without centering the features", [("c = phi - phi_bar[:, None, :]", "c = phi")]),
    ("4-trpo-step", "Fisher-vector product sums over states instead of averaging", [("c @ v) / len(phi)", "c @ v)")]),
    ("4-trpo-step", "no line search: take the full step", [("if kl(theta) <= delta and surrogate(theta) > L_old:", "if True:")]),
    ("4-trpo-step", "line search checks the improvement only", [("kl(theta) <= delta and surrogate(theta) > L_old", "surrogate(theta) > L_old")]),
    ("4-trpo-step", "line search checks the KL only", [("kl(theta) <= delta and surrogate(theta) > L_old", "kl(theta) <= delta")]),
    ("4-trpo-step", "returns the last candidate instead of theta_old", [("    return theta_old\n", "    return theta\n")]),
    ("4-trpo-step", "shrink factor ignored (always halves)", [("(shrink**k)", "(0.5**k)")]),
    ("4-trpo-step", "full step length without the 2 in beta", [("np.sqrt(2 * delta / (s @ fvp(s)))", "np.sqrt(delta / (s @ fvp(s)))")]),
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
