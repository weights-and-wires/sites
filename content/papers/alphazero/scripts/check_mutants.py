#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/alphazero/scripts/check_mutants.py
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
    ("1-az-loss", "value term not squared", [("(z - v) ** 2", "abs(z - v)")]),
    ("1-az-loss", "value sign: v - z with no square", [("(z - v) ** 2", "(v - z)")]),
    ("1-az-loss", "policy term has the wrong sign", [("-float(np.sum(pi * logp))", "float(np.sum(pi * logp))")]),
    ("1-az-loss", "L2 term dropped", [("l2 = c * float(np.sum(np.asarray(theta, dtype=float) ** 2))", "l2 = 0.0")]),
    ("1-az-loss", "L2 term is not squared", [("np.asarray(theta, dtype=float) ** 2))", "np.abs(np.asarray(theta, dtype=float))))")]),
    ("1-az-loss", "log of softmax instead of log-softmax", [("logp = _log_softmax(logits)", "e = np.exp(logits); logp = np.log(e / e.sum())")]),
    ("1-az-loss", "logit gradient is pi - p", [("d_logits = p - np.asarray(pi, dtype=float)", "d_logits = np.asarray(pi, dtype=float) - p")]),
    ("1-az-loss", "value gradient has the wrong sign", [("d_v = -2.0 * (z - v)", "d_v = 2.0 * (z - v)")]),
    ("1-az-loss", "L2 gradient misses the factor 2", [("d_theta = 2.0 * c * np.asarray(theta, dtype=float)", "d_theta = c * np.asarray(theta, dtype=float)")]),
    ("2-visit-targets", "temperature inverted (N ** tau)", [("np.log(N) / tau", "np.log(N) * tau")]),
    ("2-visit-targets", "not normalized", [("return w / w.sum()", "return w")]),
    ("2-visit-targets", "direct power overflows at small tau", [("with np.errstate(divide=\"ignore\"):\n        logits = np.log(N) / tau\n    logits = logits - logits.max()\n    w = np.exp(logits)", "w = N ** (1.0 / tau)")]),
    ("2-visit-targets", "greedy target breaks ties by index", [("return best / best.sum()", "return np.eye(len(N))[int(np.argmax(N))]")]),
    ("2-visit-targets", "tau = 0 not special-cased", [("if tau == 0:", "if False:")]),
    ("2-visit-targets", "softmax of counts instead of a power", [("np.log(N) / tau", "N / tau")]),
    ("3-root-noise", "eps and 1 - eps swapped", [("(1 - eps) * p + eps * eta", "eps * p + (1 - eps) * eta")]),
    ("3-root-noise", "noise added without mixing", [("(1 - eps) * p + eps * eta", "p + eps * eta")]),
    ("3-root-noise", "noise spread over illegal moves too", [("eta[legal] = rng.dirichlet(np.full(int(legal.sum()), alpha))", "eta[:] = rng.dirichlet(np.full(len(p), alpha))")]),
    ("3-root-noise", "alpha ignored (always 1)", [("np.full(int(legal.sum()), alpha)", "np.full(int(legal.sum()), 1.0)")]),
    ("3-root-noise", "no noise on the prior (eps ignored)", [("(1 - eps) * p + eps * eta", "0.75 * p + 0.25 * eta")]),
    ("3-root-noise", "modifies p in place", [("return (1 - eps) * p + eps * eta", "p *= (1 - eps)\n    p += eps * eta\n    return p")]),
    ("4-selfplay-tuples", "z does not alternate", [("1.0 if (T - 1 - i) % 2 == 0 else -1.0", "1.0")]),
    ("4-selfplay-tuples", "z from the first player's view", [("1.0 if (T - 1 - i) % 2 == 0 else -1.0", "1.0 if i % 2 == 0 else -1.0")]),
    ("4-selfplay-tuples", "z sign flipped", [("1.0 if (T - 1 - i) % 2 == 0 else -1.0", "-1.0 if (T - 1 - i) % 2 == 0 else 1.0")]),
    ("4-selfplay-tuples", "target is the greedy choice", [("target = visits_to_policy(N, 1.0)\n        if", "target = visits_to_policy(N, 0.0)\n        if")]),
    ("4-selfplay-tuples", "target is the prior, not the visits", [("history.append((n, target))", "history.append((n, np.asarray(prior_fn(n), dtype=float) * (np.arange(K) + 1 <= n) / (np.asarray(prior_fn(n), dtype=float) * (np.arange(K) + 1 <= n)).sum()))")]),
    ("4-selfplay-tuples", "always greedy (no sampling)", [("if len(history) < tau_moves:", "if False:")]),
    ("4-selfplay-tuples", "always samples (no greedy phase)", [("if len(history) < tau_moves:", "if True:")]),
    ("4-selfplay-tuples", "tau_moves counts from the wrong end", [("if len(history) < tau_moves:", "if len(history) >= tau_moves:")]),
    ("4-selfplay-tuples", "stores the state after the move", [("history.append((n, target))\n        n -= a + 1", "n -= a + 1\n        history.append((n, target))")]),
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
