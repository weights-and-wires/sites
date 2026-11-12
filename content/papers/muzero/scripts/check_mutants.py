#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/muzero/scripts/check_mutants.py
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
    ("1-pucb-select", "log term dropped", [("(c1 + np.log((total + c2 + 1.0) / c2))", "c1")]),
    ("1-pucb-select", "log term without the + 1", [("(total + c2 + 1.0) / c2", "(total + c2) / c2")]),
    ("1-pucb-select", "log ratio inverted", [("np.log((total + c2 + 1.0) / c2)", "np.log(c2 / (total + c2 + 1.0))")]),
    ("1-pucb-select", "log10 instead of the natural log", [("np.log((total", "np.log10((total")]),
    ("1-pucb-select", "c1 ignored", [("(c1 + np.log", "(1.0 + np.log")]),
    ("1-pucb-select", "Q is not normalized", [("q = minmax_normalize(Q, lo, hi)", "q = np.asarray(Q, dtype=float)")]),
    ("1-pucb-select", "normalized by the maximum only", [("(Q - lo) / (hi - lo)", "Q / hi")]),
    ("1-pucb-select", "flat range divides by zero", [("if hi <= lo:\n        return np.zeros_like(Q)\n", "")]),
    ("1-pucb-select", "bonus is divided by N, not 1 + N", [("(1.0 + N)", "np.maximum(N, 1e-9)")]),
    ("1-pucb-select", "square root dropped", [("np.sqrt(total)", "total")]),
    ("1-pucb-select", "square root of this edge's visits", [("np.sqrt(total)", "np.sqrt(N)")]),
    ("1-pucb-select", "prior ignored", [("P * np.sqrt(total)", "np.sqrt(total)")]),
    ("1-pucb-select", "illegal edges not masked", [("scores = np.where(np.asarray(legal, dtype=bool), scores, -np.inf)", "pass")]),
    ("1-pucb-select", "argmin instead of argmax", [("np.argmax(scores)", "np.argmin(scores)")]),
    ("1-pucb-select", "default c2 is 1", [("c2=19652.0", "c2=1.0")]),
    ("2-nstep-target", "no discount on the rewards", [("gamma ** k * u[t + k]", "u[t + k]")]),
    ("2-nstep-target", "bootstrap not discounted", [("gamma ** n * v[t + n]", "v[t + n]")]),
    ("2-nstep-target", "bootstrap discounted by gamma^(n-1)", [("gamma ** n * v[t + n]", "gamma ** (n - 1) * v[t + n]")]),
    ("2-nstep-target", "bootstraps from v[t + n - 1]", [("v[t + n]", "v[t + n - 1]")]),
    ("2-nstep-target", "sums only n - 1 rewards", [("end = min(t + n, T)", "end = min(t + n - 1, T)")]),
    ("2-nstep-target", "bootstraps past the episode end", [("if t + n < T:", "if t + n <= T:")]),
    ("2-nstep-target", "no bootstrap at all", [("if t + n < T:", "if False:")]),
    ("2-nstep-target", "reward sum runs past the end of the episode", [("end = min(t + n, T)", "end = t + n")]),
    ("3-value-transform", "linear term inside the sign factor", [("np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0) + EPS * x", "np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0 + EPS * x)")]),
    ("3-value-transform", "h without the epsilon term", [("+ EPS * x", "")]),
    ("3-value-transform", "h_inv ignores epsilon", [("return np.sign(y) * (w ** 2 - 1.0)", "return np.sign(y) * (y ** 2 + 2.0 * np.abs(y))")]),
    ("3-value-transform", "h_inv loses the sign", [("return np.sign(y) * (w ** 2 - 1.0)", "return w ** 2 - 1.0")]),
    ("3-value-transform", "phi weights swapped", [("out[low - lo] = 1.0 - frac", "out[low - lo] = frac")]),
    ("3-value-transform", "phi rounds instead of interpolating", [("frac = x - low", "frac = 0.0")]),
    ("3-value-transform", "phi does not clip", [("x = float(np.clip(x, lo, hi))", "x = float(x)")]),
    ("3-value-transform", "to_target skips the transform", [("return phi(h(x), lo, hi)", "return phi(x, lo, hi)")]),
    ("3-value-transform", "from_logits inverts each support first", [("return float(h_inv(p @ support))", "return float(p @ h_inv(support))")]),
    ("3-value-transform", "from_logits forgets to invert", [("return float(h_inv(p @ support))", "return float(p @ support)")]),
    ("3-value-transform", "softmax without the max shift", [("np.exp(logits - logits.max())", "np.exp(logits)")]),
    ("4-unrolled-loss", "reward loss at step 0", [("        if k > 0:\n            total += (u[k] - rewards[k - 1]) ** 2\n", "        total += (u[k] - (rewards[k - 1] if k > 0 else 0.0)) ** 2\n")]),
    ("4-unrolled-loss", "reward loss dropped", [("total += (u[k] - rewards[k - 1]) ** 2", "pass")]),
    ("4-unrolled-loss", "value loss dropped", [("total += (z[k] - values[k]) ** 2", "pass")]),
    ("4-unrolled-loss", "policy loss dropped", [("total += -float(np.asarray(pi[k]) @ np.log(policies[k]))", "pass")]),
    ("4-unrolled-loss", "policy loss has the wrong sign", [("total += -float(np.asarray(pi[k]) @ np.log(policies[k]))", "total += float(np.asarray(pi[k]) @ np.log(policies[k]))")]),
    ("4-unrolled-loss", "policy target is the argmax only", [("np.asarray(pi[k]) @ np.log", "np.eye(len(pi[k]))[int(np.argmax(pi[k]))] @ np.log")]),
    ("4-unrolled-loss", "reward is read from the wrong output", [("rewards.append(float(out[0]))", "rewards.append(float(out[1]))")]),
    ("4-unrolled-loss", "reward aligned to the previous step", [("u[k] - rewards[k - 1]", "u[k - 1] - rewards[k - 1]")]),
    ("4-unrolled-loss", "action ignored in the dynamics", [("np.concatenate([s, np.eye(A)[a]])", "np.concatenate([s, np.eye(A)[0]])")]),
    ("4-unrolled-loss", "actions used one step late", [("for a in actions:", "for a in ([0] + list(actions))[:len(actions)]:")]),
    ("4-unrolled-loss", "hidden state not carried forward", [("        s = out[1:]\n", "        s = H @ obs\n")]),
    ("4-unrolled-loss", "squared error without the square", [("(z[k] - values[k]) ** 2", "abs(z[k] - values[k])")]),
    ("4-unrolled-loss", "policy softmax over the wrong head", [("softmax(Wp @ s))", "softmax(wv * s))")]),
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
