#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/seq2seq/scripts/check_mutants.py
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
    ("1-source-reversal", "reverse_source reverses the input list in place", [("return list(tokens)[::-1]", "tokens.reverse()\n    return tokens")]),
    ("1-source-reversal", "lag counts the gap, not the distance (off by one)", [("return tgt_pos - src_pos", "return tgt_pos - src_pos - 1")]),
    ("1-source-reversal", "target reversed along with the source", [("tgt_pos = n + np.arange(n)", "tgt_pos = n + (n - 1 - np.arange(n) if reverse else np.arange(n))")]),
    ("1-source-reversal", "reversal never applied to the lags", [("    if reverse:\n        src_pos = n - 1 - src_pos\n", "")]),
    ("2-teacher-forced-likelihood", "decoder inputs not shifted", [("return [eos] + list(target[:-1])", "return list(target)")]),
    ("2-teacher-forced-likelihood", "decoder inputs keep the final EOS (wrong length)", [("return [eos] + list(target[:-1])", "return [eos] + list(target)")]),
    ("2-teacher-forced-likelihood", "log-softmax over the time axis", [("axis=-1", "axis=0")]),
    ("2-teacher-forced-likelihood", "log-softmax without the max subtraction", [("    z = z - z.max(axis=-1, keepdims=True)\n", "")]),
    ("2-teacher-forced-likelihood", "mean over steps instead of the sum in Equation 1", [("lp[np.arange(len(target)), np.asarray(target)].sum()", "lp[np.arange(len(target)), np.asarray(target)].mean()")]),
    ("2-teacher-forced-likelihood", "log probability taken from the previous step's row", [("lp[np.arange(len(target)), np.asarray(target)]", "lp[np.arange(len(target)) - 1, np.asarray(target)]")]),
    ("2-teacher-forced-likelihood", "perplexity without the minus sign", [("np.exp(-total_log_prob / n_tokens)", "np.exp(total_log_prob / n_tokens)")]),
    ("2-teacher-forced-likelihood", "perplexity not divided by length", [("np.exp(-total_log_prob / n_tokens)", "np.exp(-total_log_prob)")]),
    ("3-beam-search", "keeps the worst B extensions", [("cands.sort(key=lambda c: -c[1])", "cands.sort(key=lambda c: c[1])")]),
    ("3-beam-search", "EOS hypotheses stay in the beam", [("            if prefix[-1] == eos:\n                done.append((prefix, score))\n            else:\n                beam.append((prefix, score))", "            if prefix[-1] == eos:\n                done.append((prefix, score))\n            beam.append((prefix, score))")]),
    ("3-beam-search", "score is the last step only, not the sum", [("score + float(lp[tok])", "float(lp[tok])")]),
    ("3-beam-search", "beam width ignored", [("cands[:beam_size]", "cands")]),
    ("3-beam-search", "complete hypotheses not sorted", [("return sorted(done, key=lambda c: -c[1])", "return done")]),
    ("3-beam-search", "EOS hypotheses are discarded, not recorded", [("                done.append((prefix, score))\n            else:", "                pass\n            else:")]),
    ("4-fixed-vector-bottleneck", "encoder skips the recurrence A", [("h = A @ h + E[:, t]", "h = E[:, t]")]),
    ("4-fixed-vector-bottleneck", "encoder reads the tokens in reverse", [("for t in tokens:", "for t in reversed(list(tokens)):")]),
    ("4-fixed-vector-bottleneck", "readout ignores the ridge term", [("Hs.T @ Hs + ridge * np.eye(d)", "Hs.T @ Hs")]),
    ("4-fixed-vector-bottleneck", "readout reshaped as (V, n) instead of (n, V)", [("reshape(len(Hs), n, V)", "reshape(len(Hs), V, n).transpose(0, 2, 1)")]),
    ("4-fixed-vector-bottleneck", "accuracy over positions, not tokens (counts whole rows)", [("(pred == tokens).mean()", "(pred == tokens).all(axis=1).mean()")]),
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
