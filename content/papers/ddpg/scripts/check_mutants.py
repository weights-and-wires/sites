#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/ddpg/scripts/check_mutants.py
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
    ("1-critic-target", "terminal states still bootstrap", [("(1.0 - done) * ", "")]),
    ("1-critic-target", "terminal mask inverted", [("(1.0 - done) * ", "done * ")]),
    ("1-critic-target", "target actor reads the wrong state", [("actor_target(s_next)", "actor_target(r)")]),
    ("1-critic-target", "critic target reads the wrong state", [("critic_target(s_next, a_next)", "critic_target(r, a_next)")]),
    ("1-critic-target", "gamma also scales the reward", [("r + gamma * (1.0 - done) * q_next", "gamma * (r + (1.0 - done) * q_next)")]),
    ("1-critic-target", "loss sums instead of averaging", [("np.mean((y - q) ** 2)", "np.sum((y - q) ** 2)")]),
    ("1-critic-target", "loss uses absolute error", [("np.mean((y - q) ** 2)", "np.mean(np.abs(y - q))")]),
    ("2-soft-update", "tau and 1 - tau swapped", [("tau * source[k] + (1.0 - tau) * target[k]", "(1.0 - tau) * source[k] + tau * target[k]")]),
    ("2-soft-update", "hard copy", [("tau * source[k] + (1.0 - tau) * target[k]", "source[k].copy()")]),
    ("2-soft-update", "updates the target in place", [("return {k: tau * source[k] + (1.0 - tau) * target[k] for k in target}", "for k in target:\n        target[k] *= (1.0 - tau)\n        target[k] += tau * source[k]\n    return target")]),
    ("2-soft-update", "returns the source arrays without copying", [("return {k: tau * source[k] + (1.0 - tau) * target[k] for k in target}", "return {k: source[k] if tau == 1.0 else tau * source[k] + (1.0 - tau) * target[k] for k in target}")]),
    ("2-soft-update", "rounds down instead of up", [("math.ceil(", "math.floor(")]),
    ("2-soft-update", "uses tau instead of 1 - tau in the logarithm", [("math.log(1.0 - tau)", "math.log(tau)")]),
    ("3-ou-noise", "pull moves away from mu", [("self.theta * (self.mu - self.x) * self.dt", "self.theta * (self.x - self.mu) * self.dt")]),
    ("3-ou-noise", "noise scaled by dt instead of sqrt(dt)", [("self.sigma * np.sqrt(self.dt) * eps", "self.sigma * self.dt * eps")]),
    ("3-ou-noise", "independent noise, no memory", [("self.x = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * eps", "self.x = self.mu + self.sigma * np.sqrt(self.dt) * eps")]),
    ("3-ou-noise", "returns the live state, not a copy", [("return self.x.copy()", "return self.x")]),
    ("3-ou-noise", "reset does nothing", [("        self.x = np.full(self.dim, self.mu, dtype=float)\n\n    def sample", "        pass\n\n    def sample")]),
    ("3-ou-noise", "pull ignores dt", [("self.theta * (self.mu - self.x) * self.dt", "self.theta * (self.mu - self.x)")]),
    ("4-policy-gradient", "missing the actor Jacobian factor", [("dq_da[:, None] * S", "dq_da[:, None] * np.ones_like(S)")]),
    ("4-policy-gradient", "wrong sign", [("dq_da = -2.0 * (a - S @ k)", "dq_da = 2.0 * (a - S @ k)")]),
    ("4-policy-gradient", "sum instead of mean", [("np.mean(dq_da[:, None] * S, axis=0)", "np.sum(dq_da[:, None] * S, axis=0)")]),
    ("4-policy-gradient", "factor 2 dropped", [("dq_da = -2.0 * (a - S @ k)", "dq_da = -1.0 * (a - S @ k)")]),
    ("4-policy-gradient", "gradient descent instead of ascent", [("theta = theta + lr", "theta = theta - lr")]),
    ("4-policy-gradient", "ascend takes one step only", [("for _ in range(steps):", "for _ in range(min(steps, 1)):")]),
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
