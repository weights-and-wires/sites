#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/dapo/scripts/check_mutants.py
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
    ("1-clip-higher", "symmetric clip (eps_high ignored)", [("1 + eps_high)\n    return np.minimum(ratio * adv", "1 + eps_low)\n    return np.minimum(ratio * adv")]),
    ("1-clip-higher", "swapped bounds", [("np.clip(ratio, 1 - eps_low, 1 + eps_high)", "np.clip(ratio, 1 - eps_high, 1 + eps_low)")]),
    ("1-clip-higher", "max instead of min", [("np.minimum(ratio * adv, clipped * adv)", "np.maximum(ratio * adv, clipped * adv)")]),
    ("1-clip-higher", "clipped term only (no min with r * A)", [("np.minimum(ratio * adv, clipped * adv)", "clipped * adv")]),
    ("1-clip-higher", "ceiling not capped at 1", [("np.minimum(1.0, np.asarray(p_old, dtype=float) * (1 + eps_high))", "np.asarray(p_old, dtype=float) * (1 + eps_high)")]),
    ("1-clip-higher", "ceiling uses eps instead of 1 + eps", [("* (1 + eps_high))", "* eps_high)")]),
    ("2-token-level-loss", "token level divides by G", [("sum(np.sum(t) for t in terms) / sum(len(t) for t in terms)", "sum(np.sum(t) for t in terms) / len(terms)")]),
    ("2-token-level-loss", "sample level is the token mean", [('return float(np.mean([np.mean(t) for t in terms]))', 'return float(sum(np.sum(t) for t in terms) / sum(len(t) for t in terms))')]),
    ("2-token-level-loss", "sample level sums instead of averaging tokens", [("np.mean([np.mean(t) for t in terms])", "np.mean([np.sum(t) for t in terms])")]),
    ("2-token-level-loss", "sample weights miss the 1/G", [("1.0 / (g * n)", "1.0 / n")]),
    ("2-token-level-loss", "token weights use the output length", [("np.full(n, 1.0 / total)", "np.full(n, 1.0 / n)")]),
    ("3-dynamic-sampling", "advantages ignore the std", [("return (r - r.mean()) / std", "return r - r.mean()")]),
    ("3-dynamic-sampling", "sample std (ddof=1)", [("std = r.std()", "std = r.std(ddof=1)")]),
    ("3-dynamic-sampling", "no zero-std guard", [("if std == 0:", "if False:")]),
    ("3-dynamic-sampling", "keeps all-correct groups", [("0 < correct < len(r)", "0 < correct <= len(r)")]),
    ("3-dynamic-sampling", "keeps all-wrong groups", [("0 < correct < len(r)", "correct < len(r)")]),
    ("3-dynamic-sampling", "does not stop when full", [("            if len(kept) == batch_size:\n                break\n", "")]),
    ("3-dynamic-sampling", "counts only kept groups as drawn", [("    for g in groups:\n        drawn += 1\n        if keep_group(g):\n            kept.append(g)\n", "    for g in groups:\n        if keep_group(g):\n            drawn += 1\n            kept.append(g)\n")]),
    ("4-overlong-shaping", "ramp starts at l_max", [("if length <= l_max - l_cache:", "if length <= l_max:")]),
    ("4-overlong-shaping", "ramp is positive", [("((l_max - l_cache) - length) / l_cache", "(length - (l_max - l_cache)) / l_cache")]),
    ("4-overlong-shaping", "ramp divides by l_max", [("- length) / l_cache", "- length) / l_max")]),
    ("4-overlong-shaping", "no -1 beyond the limit", [("    return -1.0\n\n\ndef total_reward", "    return ((l_max - l_cache) - length) / l_cache\n\n\ndef total_reward")]),
    ("4-overlong-shaping", "wrong reward for incorrect (0 not -1)", [("1.0 if correct else -1.0", "1.0 if correct else 0.0")]),
    ("4-overlong-shaping", "total ignores the penalty", [("return base + soft_overlong_penalty(length, l_max, l_cache)", "return base")]),
    ("4-overlong-shaping", "total drops custom constants", [("soft_overlong_penalty(length, l_max, l_cache)", "soft_overlong_penalty(length)")]),
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
