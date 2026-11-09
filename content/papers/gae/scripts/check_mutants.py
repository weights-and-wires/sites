#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/gae/scripts/check_mutants.py
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
    ("1-gae-estimator", "decay is lambda alone, without the discount", [("acc = delta[t] + gamma * lam * acc", "acc = delta[t] + lam * acc")]),
    ("1-gae-estimator", "decay is gamma alone, without lambda", [("acc = delta[t] + gamma * lam * acc", "acc = delta[t] + gamma * acc")]),
    ("1-gae-estimator", "residual forgets to subtract V(s_t)", [("return rewards + gamma * next_values - values", "return rewards + gamma * next_values")]),
    ("1-gae-estimator", "residual ignores the bootstrap value", [("np.append(values[1:], last_value)", "np.append(values[1:], 0.0)")]),
    ("1-gae-estimator", "residual uses V(s_t) in place of V(s_{t+1})", [("np.append(values[1:], last_value)", "values")]),
    ("1-gae-estimator", "forward pass: advantage sums over the past", [("for t in reversed(range(len(delta))):", "for t in range(len(delta)):")]),
    ("1-gae-estimator", "residual drops the discount on V(s_{t+1})", [("rewards + gamma * next_values - values", "rewards + next_values - values")]),
    ("2-rollout-buffer", "done masks the bootstrap but not the carry", [("acc = delta + gamma * lam * mask * acc", "acc = delta + gamma * lam * acc")]),
    ("2-rollout-buffer", "done masks the carry but not the bootstrap", [("gamma * next_value * mask", "gamma * next_value")]),
    ("2-rollout-buffer", "last_value ignored for a cut-off episode", [("last_value if t == n - 1 else values[t + 1]", "0.0 if t == n - 1 else values[t + 1]")]),
    ("2-rollout-buffer", "last_value used even after a done step", [("gamma * next_value * mask - values[t]", "gamma * (next_value if t < n - 1 else last_value) - values[t]")]),
    ("2-rollout-buffer", "mask is inverted (done = 0 means ended)", [("mask = 1.0 - dones[t]", "mask = dones[t]")]),
    ("3-k-step-mixture", "mixture leaves out the leftover weight", [("    total += lam ** (n - 1) * k_step_advantage(rewards, values, gamma, t, n)\n", "")]),
    ("3-k-step-mixture", "weights are lambda^k instead of lambda^(k-1)", [("(1 - lam) * lam ** (k - 1) *", "(1 - lam) * lam**k *")]),
    ("3-k-step-mixture", "k-step estimate forgets to subtract V(s_t)", [("total = -values[t]", "total = 0.0")]),
    ("3-k-step-mixture", "k-step bootstrap discount is gamma^(k-1)", [("total += gamma**k * values[t + k]", "total += gamma ** (k - 1) * values[t + k]")]),
    ("3-k-step-mixture", "k-step reward sum is undiscounted", [("total += gamma**l * rewards[t + l]", "total += rewards[t + l]")]),
    ("3-k-step-mixture", "k is not capped at the end of the episode", [("    k = min(k, T - t)\n", "")]),
    ("4-shaped-rewards", "discounted sum runs forward in time", [("for t in reversed(range(len(x))):", "for t in range(len(x)):")]),
    ("4-shaped-rewards", "shaping forgets the discount on Phi(s')", [("rewards + gamma * next_phi - phi", "rewards + next_phi - phi")]),
    ("4-shaped-rewards", "shaping adds Phi(s) instead of subtracting it", [("rewards + gamma * next_phi - phi", "rewards + gamma * next_phi + phi")]),
    ("4-shaped-rewards", "shaping ignores last_phi", [("np.append(phi[1:], last_phi)", "np.append(phi[1:], 0.0)")]),
    ("4-shaped-rewards", "GAE sums with gamma instead of gamma * lambda", [("discounted_cumsum(shaped, gamma * lam)", "discounted_cumsum(shaped, gamma)")]),
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
