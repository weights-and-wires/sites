#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/dqn/scripts/check_mutants.py
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
    ("1-td-target", "terminal states still bootstrap", [("(1.0 - done) * ", "")]),
    ("1-td-target", "max over the batch axis", [("np.max(q_next, axis=1)", "np.max(q_next, axis=0)")]),
    ("1-td-target", "min instead of max over next actions", [("np.max(q_next, axis=1)", "np.min(q_next, axis=1)")]),
    ("1-td-target", "gamma also scales the reward", [("r + gamma * (1.0 - done) * np.max(q_next, axis=1)", "gamma * (r + (1.0 - done) * np.max(q_next, axis=1))")]),
    ("1-td-target", "loss uses the best action, not the taken one", [("q[np.arange(len(a)), a]", "q.max(axis=1)")]),
    ("1-td-target", "loss sums instead of averaging", [("np.mean((y - q_taken) ** 2)", "np.sum((y - q_taken) ** 2)")]),
    ("2-replay-buffer", "pointer stops advancing when full (overwrites the newest slot)", [("self.pos = (self.pos + 1) % self.capacity", "self.pos = min(self.pos + 1, self.capacity - 1)")]),
    ("2-replay-buffer", "samples from the whole capacity, including empty slots", [("rng.integers(len(self), size=batch_size)", "rng.integers(self.capacity, size=batch_size)")]),
    ("2-replay-buffer", "next states drawn with a separate index", [("self.S2[idx]", "self.S2[rng.integers(len(self), size=batch_size)]")]),
    ("2-replay-buffer", "samples only the older half", [("rng.integers(len(self), size=batch_size)", "rng.integers(max(1, len(self) // 2), size=batch_size)")]),
    ("3-epsilon-greedy", "epsilon keeps falling below the floor", [("return max(eps_end, eps_start + (eps_end - eps_start) * frame / anneal_frames)", "return eps_start + (eps_end - eps_start) * frame / anneal_frames")]),
    ("3-epsilon-greedy", "random branch excludes the greedy action", [("return int(rng.integers(len(q_values)))", "g = int(np.argmax(q_values))\n        return int((g + 1 + rng.integers(len(q_values) - 1)) % len(q_values))")]),
    ("3-epsilon-greedy", "explore and exploit swapped", [("rng.random() < epsilon", "rng.random() > epsilon")]),
    ("3-epsilon-greedy", "argmin instead of argmax", [("np.argmax(q_values)", "np.argmin(q_values)")]),
    ("3-epsilon-greedy", "anneals twice as fast", [("/ anneal_frames)", "/ (anneal_frames / 2))")]),
    ("4-tiny-dqn", "terminal states still bootstrap", [("(1.0 - D) * ", "")]),
    ("4-tiny-dqn", "min instead of max in the target", [("q_values(p, S2).max(axis=1)", "q_values(p, S2).min(axis=1)")]),
    ("4-tiny-dqn", "error spread over every action output", [("dq[idx, A] = 2 * err / B", "dq[:] = (2 * err / B)[:, None]")]),
    ("4-tiny-dqn", "missing ReLU derivative", [(" * (z1 > 0)", "")]),
    ("4-tiny-dqn", "gradient ascent instead of descent", [("p[k] -= lr * g", "p[k] += lr * g")]),
    ("4-tiny-dqn", "sum instead of mean over the batch", [("2 * err / B", "2 * err")]),
    ("4-tiny-dqn", "W2 updated before backpropagating through it", [("    dh = (dq @ p[\"W2\"].T) * (z1 > 0)\n", "    p[\"W2\"] -= lr * (h.T @ dq)\n    dh = (dq @ p[\"W2\"].T) * (z1 > 0)\n")]),
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
