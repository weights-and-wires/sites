#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/prm/scripts/check_mutants.py
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
    ("1-solution-score", "neutral counted as negative by default", [('if neutral == "positive":\n        return p_pos + p_neu', 'if neutral == "negative":\n        return p_pos + p_neu')]),
    ("1-solution-score", "neutral ignored", [("return p_pos + p_neu", "return p_pos.copy()")]),
    ("1-solution-score", "product replaced by mean", [("np.prod(scores)", "np.mean(scores)")]),
    ("1-solution-score", "min replaced by max", [("np.min(scores)", "np.max(scores)")]),
    ("1-solution-score", "min and product swapped", [('if reduction == "product":', 'if reduction == "min":')]),
    ("1-solution-score", "empty solution allowed", [("if scores.size == 0:", "if False:")]),
    ("1-solution-score", "bad reduction falls back to product", [('    if reduction == "min":\n        return float(np.min(scores))\n    raise', '    if reduction == "min":\n        return float(np.min(scores))\n    return float(np.prod(scores))\n    raise')]),
    ("1-solution-score", "modifies p_pos in place", [("return p_pos + p_neu", "p_pos += p_neu\n        return p_pos")]),
    ("2-best-of-n-and-voting", "best_of_n returns the lowest score", [("if s > scores[best]:", "if s < scores[best]:")]),
    ("2-best-of-n-and-voting", "best_of_n tie goes to the last", [("if s > scores[best]:", "if s >= scores[best]:")]),
    ("2-best-of-n-and-voting", "best_of_n returns the value", [("    return best\n\n\ndef majority_vote", "    return scores[best]\n\n\ndef majority_vote")]),
    ("2-best-of-n-and-voting", "majority vote picks the largest label", [("return Counter(answers).most_common(1)[0][0]", "return max(answers)")]),
    ("2-best-of-n-and-voting", "majority vote tie goes to the last", [("return Counter(answers).most_common(1)[0][0]", "return Counter(reversed(answers)).most_common(1)[0][0]")]),
    ("2-best-of-n-and-voting", "weighted vote takes the max score", [("totals.get(a, 0.0) + s", "max(totals.get(a, 0.0), s)")]),
    ("2-best-of-n-and-voting", "weighted vote tie goes to the last", [("if best_total is None or t > best_total:", "if best_total is None or t >= best_total:")]),
    ("2-best-of-n-and-voting", "weighted vote ignores the scores", [("totals.get(a, 0.0) + s", "totals.get(a, 0.0) + 1")]),
    ("3-step-labels", "process labels run past the first error", [("        if not ok:\n            break\n", "")]),
    ("3-step-labels", "process labels drop the incorrect step", [("    for ok in step_ok:\n        labels.append(1 if ok else 0)\n        if not ok:\n            break", "    for ok in step_ok:\n        if not ok:\n            break\n        labels.append(1)")]),
    ("3-step-labels", "outcome labels inverted", [("[1 if final_correct else 0] * n_steps", "[0 if final_correct else 1] * n_steps")]),
    ("3-step-labels", "outcome labels have one entry", [("[1 if final_correct else 0] * n_steps", "[1 if final_correct else 0]")]),
    ("3-step-labels", "threshold is not strict", [("p <= threshold", "p < threshold")]),
    ("3-step-labels", "threshold ignored", [("p <= threshold", "p <= 0.2")]),
    ("3-step-labels", "comparison inverted", [("p <= threshold", "p >= threshold")]),
    ("4-best-of-n-simulation", "argmin instead of argmax", [("scores.argmax(axis=1)", "scores.argmin(axis=1)")]),
    ("4-best-of-n-simulation", "noise ignored", [("sigma * rng.standard_normal((trials, n))", "0")]),
    ("4-best-of-n-simulation", "scorer ignores correctness", [("sep * correct +", "0 * correct +")]),
    ("4-best-of-n-simulation", "one draw shared by all n", [("rng.random((trials, n)) < p\n    scores", "np.repeat(rng.random((trials, 1)), n, axis=1) < p\n    scores")]),
    ("4-best-of-n-simulation", "accuracy of the first sample", [("correct[np.arange(trials), pick].mean()", "correct[:, 0].mean()")]),
    ("4-best-of-n-simulation", "vote counts the wrong answer as 0", [("answers = np.where(correct, 0, wrong)", "answers = np.where(correct, wrong, 0)")]),
    ("4-best-of-n-simulation", "vote tie goes to the correct answer", [("Counter(row).most_common(1)[0][0] == 0", "Counter(row)[0] >= max(Counter(row).values())")]),
    ("4-best-of-n-simulation", "vote tie goes to a wrong answer", [("Counter(row).most_common(1)[0][0] == 0", "Counter(row)[0] > max(c for a, c in Counter(row).items() if a != 0) if len(set(row)) > 1 else True")]),
    ("4-best-of-n-simulation", "vote uses any correct sample", [("Counter(row).most_common(1)[0][0] == 0", "0 in row")]),
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
