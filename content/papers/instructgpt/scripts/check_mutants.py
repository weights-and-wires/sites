#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/instructgpt/scripts/check_mutants.py [--skip-torch]
"""

import importlib.util
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
    ("1-ranking-loss", "winner and loser swapped in the difference", [("rewards[ranking[i]] - rewards[ranking[j]]", "rewards[ranking[j]] - rewards[ranking[i]]")]),
    ("1-ranking-loss", "sum over pairs instead of the mean", [("np.mean(losses)", "np.sum(losses)")]),
    ("1-ranking-loss", "only adjacent pairs in the ranking", [("for j in range(i + 1, len(ranking)):", "for j in range(i + 1, min(i + 2, len(ranking))):")]),
    ("1-ranking-loss", "log(sigmoid(d)) computed with exp, overflows", [("np.logaddexp(0.0, -diff)", "-np.log(1.0 / (1.0 + np.exp(-diff)))")]),
    ("1-ranking-loss", "divide by K instead of the number of pairs", [("np.mean(losses)", "np.sum(losses) / len(ranking)")]),
    ("2-kl-token-rewards", "RM score added to every token", [("r[-1] += rm_score", "r += rm_score")]),
    ("2-kl-token-rewards", "RM score added to the first token", [("r[-1] += rm_score", "r[0] += rm_score")]),
    ("2-kl-token-rewards", "penalty sign flipped (rewards drifting from SFT)", [("r = -beta *", "r = beta *")]),
    ("2-kl-token-rewards", "probability ratio instead of log ratio", [("(np.asarray(logp_rl, dtype=float) - np.asarray(logp_sft, dtype=float))", "np.exp(np.asarray(logp_rl, dtype=float) - np.asarray(logp_sft, dtype=float))")]),
    ("2-kl-token-rewards", "writes into the caller's array", [("    r = -beta * (np.asarray(logp_rl, dtype=float) - np.asarray(logp_sft, dtype=float))\n    r[-1] += rm_score", "    r = np.asarray(logp_rl, dtype=float)\n    r -= np.asarray(logp_sft, dtype=float)\n    r *= -beta\n    r[-1] += rm_score")]),
    ("3-ppo-ptx-objective", "log ratio averaged over tokens instead of summed", [(".sum(axis=1)", ".mean(axis=1)")]),
    ("3-ppo-ptx-objective", "pretraining term subtracted", [("rl_term + gamma * pre_term", "rl_term - gamma * pre_term")]),
    ("3-ppo-ptx-objective", "gamma not applied", [("rl_term + gamma * pre_term", "rl_term + pre_term")]),
    ("3-ppo-ptx-objective", "pretraining term summed instead of averaged", [("np.mean(np.asarray(pre_logp, dtype=float))", "np.sum(np.asarray(pre_logp, dtype=float))")]),
    ("3-ppo-ptx-objective", "KL penalty added instead of subtracted", [("rm_scores, dtype=float) - beta * log_ratio", "rm_scores, dtype=float) + beta * log_ratio")]),
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
    skip_torch = "--skip-torch" in sys.argv or importlib.util.find_spec("torch") is None
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
