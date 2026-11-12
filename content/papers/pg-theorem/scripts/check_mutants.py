#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/pg-theorem/scripts/check_mutants.py
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
    ("1-exact-policy-gradient", "d uses only the start state (drops the state distribution)", [("return np.linalg.solve(np.eye(P.shape[0]) - gamma * p_pi.T, e0)", "return e0")]),
    ("1-exact-policy-gradient", "d normalized to sum to 1", [("return np.linalg.solve(np.eye(P.shape[0]) - gamma * p_pi.T, e0)", "d = np.linalg.solve(np.eye(P.shape[0]) - gamma * p_pi.T, e0)\n    return d / d.sum()")]),
    ("1-exact-policy-gradient", "d solved with P instead of its transpose", [("np.eye(P.shape[0]) - gamma * p_pi.T, e0", "np.eye(P.shape[0]) - gamma * p_pi, e0")]),
    ("1-exact-policy-gradient", "gradient uses the same weight 1 for every state", [("return d[:, None] * pi * (q - v)", "return pi * (q - v)")]),
    ("1-exact-policy-gradient", "gradient uses reward R in place of Q", [("q = action_values(P, R, pi, gamma)\n    d", "q = R\n    d")]),
    ("1-exact-policy-gradient", "Q forgets to discount the next value", [("return R + gamma * np.einsum(\"sat,t->sa\", P, v)", "return R + np.einsum(\"sat,t->sa\", P, v)")]),
    ("1-exact-policy-gradient", "V solved with the transposed transition matrix", [("np.eye(P.shape[0]) - gamma * p_pi, r_pi", "np.eye(P.shape[0]) - gamma * p_pi.T, r_pi")]),
    ("1-exact-policy-gradient", "gradient has the wrong sign", [("return d[:, None] * pi * (q - v)", "return -d[:, None] * pi * (q - v)")]),
    ("2-compatible-critic", "features not centered (raw phi)", [("return phi - np.einsum(\"sb,sbk->sk\", pi, phi)[:, None, :]", "return phi + 0.0")]),
    ("2-compatible-critic", "mean over actions is uniform, not pi", [("np.einsum(\"sb,sbk->sk\", pi, phi)[:, None, :]", "phi.mean(axis=1)[:, None, :]")]),
    ("2-compatible-critic", "fit ignores the state weights d", [("wts = d[:, None] * pi\n", "wts = pi\n")]),
    ("2-compatible-critic", "fit ignores the policy weights pi", [("wts = d[:, None] * pi\n", "wts = d[:, None] * np.ones_like(pi)\n")]),
    ("2-compatible-critic", "gradient uses Q-less weights d only", [("np.einsum(\"sa,sak,sa->k\", d[:, None] * pi, psi, psi @ w)", "np.einsum(\"sa,sak,sa->k\", d[:, None] * np.ones_like(pi), psi, psi @ w)")]),
    ("2-compatible-critic", "gradient drops the critic (uses 1 for f_w)", [("psi @ w)", "np.ones(psi.shape[:2]))")]),
    ("2-compatible-critic", "least squares on psi only, no weights at all", [("wts = d[:, None] * pi\n", "wts = np.ones_like(pi)\n")]),
    ("3-reinforce", "no gamma^t factor", [("coef = gamma**t * ret[:, t]", "coef = ret[:, t]")]),
    ("3-reinforce", "total return instead of reward-to-go", [("coef = gamma**t * ret[:, t]", "coef = gamma**t * ret[:, 0]")]),
    ("3-reinforce", "one-hot only, no subtraction of pi", [("            np.add.at(grad[:, b], s, -coef * pi[s, b])", "            pass")]),
    ("3-reinforce", "sum over episodes instead of mean", [("return grad / n", "return grad")]),
    ("3-reinforce", "reward-to-go without discounting", [("acc = rewards[:, t] + gamma * acc", "acc = rewards[:, t] + acc")]),
    ("3-reinforce", "ascent sign flipped", [("return grad / n", "return -grad / n")]),
    ("3-reinforce", "pi row indexed by the action instead of the state", [("pi[s, b]", "pi[a, b]")]),
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
