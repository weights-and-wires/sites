#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/npg/scripts/check_mutants.py
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
    ("1-softmax-fisher", "tabular F without the outer product term", [("np.diag(p) - np.outer(p, p)", "np.diag(p)")]),
    ("1-softmax-fisher", "tabular F with the sign flipped", [("np.diag(p) - np.outer(p, p)", "np.outer(p, p) - np.diag(p)")]),
    ("1-softmax-fisher", "features not centered", [("psi = phi - np.einsum(\"sk,skd->sd\", pi, phi)[:, None, :]", "psi = phi")]),
    ("1-softmax-fisher", "no state distribution weight", [('np.einsum("s,sk,ska,skb->ab", rho, pi, psi, psi)', 'np.einsum("s,sk,ska,skb->ab", np.ones_like(rho), pi, psi, psi)')]),
    ("1-softmax-fisher", "uniform weight over actions instead of pi", [('np.einsum("s,sk,ska,skb->ab", rho, pi, psi, psi)', 'np.einsum("s,sk,ska,skb->ab", rho, np.ones_like(pi) / pi.shape[1], psi, psi)')]),
    ("1-softmax-fisher", "rho normalized by the number of states", [("return np.einsum(\"s,sk,ska,skb->ab\", rho, pi, psi, psi)", "return np.einsum(\"s,sk,ska,skb->ab\", rho, pi, psi, psi) / len(rho)")]),
    ("2-natural-gradient", "policy gradient uses phi instead of psi", [("psi = scores(phi, theta)\n    return np.einsum(\"s,sk,skd,sk->d\", rho, pi, psi, Q)", "return np.einsum(\"s,sk,skd,sk->d\", rho, pi, phi, Q)")]),
    ("2-natural-gradient", "policy gradient without the state distribution", [('np.einsum("s,sk,skd,sk->d", rho, pi, psi, Q)', 'np.einsum("s,sk,skd,sk->d", np.ones_like(rho), pi, psi, Q)')]),
    ("2-natural-gradient", "policy gradient without the pi weight", [('np.einsum("s,sk,skd,sk->d", rho, pi, psi, Q)', 'np.einsum("s,sk,skd,sk->d", rho, np.ones_like(pi), psi, Q)')]),
    ("2-natural-gradient", "damping ignored", [("F + lam * np.eye(len(g))", "F")]),
    ("2-natural-gradient", "damping added to every entry", [("F + lam * np.eye(len(g))", "F + lam")]),
    ("2-natural-gradient", "damping multiplies F", [("F + lam * np.eye(len(g))", "F * (1 + lam)")]),
    ("2-natural-gradient", "returns the plain gradient", [("return np.linalg.solve(F + lam * np.eye(len(g)), g)", "return g")]),
    ("2-natural-gradient", "multiplies by F instead of solving", [("np.linalg.solve(F + lam * np.eye(len(g)), g)", "(F + lam * np.eye(len(g))) @ g")]),
    ("2-natural-gradient", "changes the default damping", [("lam=1e-3", "lam=1e-1")]),
    ("3-reparameterisation", "result left in u coordinates", [("return c * delta_u", "return delta_u")]),
    ("3-reparameterisation", "features scaled the wrong way", [("phi_u = phi * c", "phi_u = phi / c")]),
    ("3-reparameterisation", "natural step uses the Fisher matrix of the original coordinates", [("fisher(phi_u, u, rho)", "fisher(phi, theta, rho)")]),
    ("3-reparameterisation", "gradient taken in the original coordinates", [("g_u = policy_gradient(phi_u, u, rho, Q)", "g_u = policy_gradient(phi, theta, rho, Q)")]),
    ("3-reparameterisation", "natural flag ignored", [("    if natural:\n", "    if False:\n")]),
    ("4-compatible-weights", "unweighted fit", [("sw = np.sqrt((rho[:, None] * pi).reshape(S * K))", "sw = np.ones(S * K)")]),
    ("4-compatible-weights", "weights without the state distribution", [("sw = np.sqrt((rho[:, None] * pi).reshape(S * K))", "sw = np.sqrt(pi.reshape(S * K))")]),
    ("4-compatible-weights", "weights without the policy probability", [("sw = np.sqrt((rho[:, None] * pi).reshape(S * K))", "sw = np.sqrt(np.repeat(rho, K))")]),
    ("4-compatible-weights", "weights not square-rooted", [("sw = np.sqrt((rho[:, None] * pi).reshape(S * K))", "sw = (rho[:, None] * pi).reshape(S * K)")]),
    ("4-compatible-weights", "fit on the raw features phi", [("X = scores(phi, theta).reshape(S * K, d)", "X = phi.reshape(S * K, d)")]),
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
