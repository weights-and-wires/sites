#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/snell/scripts/check_mutants.py
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
    ("1-weighted-selection", "last step replaced by first step", [("float(step_scores[-1])", "float(step_scores[0])")]),
    ("1-weighted-selection", "prod replaced by sum", [("float(np.prod(step_scores))", "float(np.sum(step_scores))")]),
    ("1-weighted-selection", "min replaced by max", [("float(np.min(step_scores))", "float(np.max(step_scores))")]),
    ("1-weighted-selection", "best_of_n returns the index", [("return answers[int(np.argmax(scores))]", "return int(np.argmax(scores))")]),
    ("1-weighted-selection", "best_of_n takes the lowest score", [("np.argmax(scores)", "np.argmin(scores)")]),
    ("1-weighted-selection", "weighted uses the max score", [("totals[a] = totals.get(a, 0.0) + float(s)\n    else", "totals[a] = max(totals.get(a, 0.0), float(s))\n    else")]),
    ("1-weighted-selection", "weighted counts votes (ignores scores)", [("totals.get(a, 0.0) + float(s)", "totals.get(a, 0.0) + 1.0")]),
    ("1-weighted-selection", "majority sums the scores", [("totals[a] = totals.get(a, 0) + 1", "totals[a] = totals.get(a, 0) + 0.5 * (len(answers) - 2 * totals.get(a, 0))")]),
    ("1-weighted-selection", "ties go to the last answer", [("if best is None or t > totals[best]:", "if best is None or t >= totals[best]:")]),
    ("1-weighted-selection", "unknown method falls through silently", [("raise ValueError(f\"unknown method: {method!r}\")", "totals = {}")]),
    ("2-beam-search", "keeps the worst prefixes", [("sorted(beams, key=lambda p: -score(p))\n        kept", "sorted(beams, key=lambda p: score(p))\n        kept")]),
    ("2-beam-search", "keeps N instead of N/M", [("ranked[: n_beams // beam_width]", "ranked[:n_beams]")]),
    ("2-beam-search", "expands with N/M children", [("expand(prefix, beam_width)", "expand(prefix, n_beams // beam_width)")]),
    ("2-beam-search", "keeps one extra prefix", [("ranked[: n_beams // beam_width]", "ranked[: n_beams // beam_width + 1]")]),
    ("2-beam-search", "one round too many", [("range(n_steps - 1)", "range(n_steps)")]),
    ("2-beam-search", "one round too few", [("range(n_steps - 1)", "range(max(n_steps - 2, 0))")]),
    ("2-beam-search", "final list is not sorted", [("return sorted(beams, key=lambda p: -score(p))", "return beams")]),
    ("2-beam-search", "final list sorted worst first", [("return sorted(beams, key=lambda p: -score(p))", "return sorted(beams, key=lambda p: score(p))")]),
    ("3-difficulty-bins", "bin 0 is the hardest", [("np.argsort(-pass1, kind=\"stable\")", "np.argsort(pass1, kind=\"stable\")")]),
    ("3-difficulty-bins", "equal-width ranges instead of quantiles", [("bins[order] = (np.arange(n) * n_bins) // n", "bins[:] = np.minimum((np.clip(1 - pass1, 0, 0.999999) * n_bins).astype(int), n_bins - 1)")]),
    ("3-difficulty-bins", "ties broken by the higher index", [("np.argsort(-pass1, kind=\"stable\")", "np.argsort(-pass1[::-1], kind=\"stable\")[::-1]")]),
    ("3-difficulty-bins", "default bin count of 4", [("def difficulty_bins(pass1, n_bins=5)", "def difficulty_bins(pass1, n_bins=4)")]),
    ("3-difficulty-bins", "picks the worst strategy per bin", [("if mean > best_mean:", "if mean < best_mean or best_name is None:")]),
    ("3-difficulty-bins", "ties go to the last strategy", [("if mean > best_mean:", "if mean >= best_mean:")]),
    ("3-difficulty-bins", "accuracy averages the bin means", [("return choices, n_correct / len(bins)", "return choices, sum(float(np.mean(np.asarray(outcomes[c])[bins == b])) for b, c in enumerate(choices)) / n_bins")]),
    ("4-flops-matched", "pretraining uses 2 N D", [("return 6 * n_params * d_tokens", "return 2 * n_params * d_tokens")]),
    ("4-flops-matched", "inference uses 6 N D", [("return 2 * n_params * d_tokens", "return 6 * n_params * d_tokens")]),
    ("4-flops-matched", "R inverted", [("3 * (m - 1) / r", "3 * (m - 1) * r")]),
    ("4-flops-matched", "factor 3 dropped", [("3 * (m - 1) / r", "(m - 1) / r")]),
    ("4-flops-matched", "pretraining cost ignored", [("m + 3 * (m - 1) / r", "m")]),
    ("4-flops-matched", "uses M instead of M - 1", [("3 * (m - 1) / r", "3 * m / r")]),
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
