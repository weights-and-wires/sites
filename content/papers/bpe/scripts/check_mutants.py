#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/bpe/scripts/check_mutants.py
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

MERGE_LINE = "out[pattern.sub(\"\".join(pair), word)] = freq"

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-pair-counts-and-merge", "count each pair once per word, ignoring frequency", [("pairs[symbols[i], symbols[i + 1]] += freq", "pairs[symbols[i], symbols[i + 1]] += 1")]),
    ("1-pair-counts-and-merge", "skip the pair that ends in </w>", [("for i in range(len(symbols) - 1):", "for i in range(len(symbols) - 2):")]),
    ("1-pair-counts-and-merge", "naive str.replace instead of whole-symbol match", [
        ("    out = {}\n    bigram", "    out = {}\n    for word, freq in vocab.items():\n        out[word.replace(\" \".join(pair), \"\".join(pair))] = freq\n    return out\n    bigram")]),
    ("1-pair-counts-and-merge", "pattern without the lookbehind", [("(?<!\\S)\" + bigram", "\" + bigram")]),
    ("1-pair-counts-and-merge", "merge changes the input dict", [("    out = {}\n    bigram", "    out = vocab\n    bigram"), ("out[pattern.sub(\"\".join(pair), word)] = freq", "out[pattern.sub(\"\".join(pair), word)] = freq; vocab.pop(word) if pattern.sub(\"\".join(pair), word) != word else None")]),
    ("1-pair-counts-and-merge", "merge drops frequency", [(MERGE_LINE, "out[pattern.sub(\"\".join(pair), word)] = 1")]),
    ("2-learn-and-encode", "learner ignores word frequency", [("vocab = {\" \".join(w) + \" </w>\": f for w, f in word_freqs.items()}", "vocab = {\" \".join(w) + \" </w>\": 1 for w, f in word_freqs.items()}")]),
    ("2-learn-and-encode", "ties go to the largest pair", [("key=lambda p: (-pairs[p], p)", "key=lambda p: (pairs[p], p)")]),
    ("2-learn-and-encode", "no end-of-word symbol in the learner", [("\" \".join(w) + \" </w>\"", "\" \".join(w)")]),
    ("2-learn-and-encode", "no end-of-word symbol in the encoder", [("symbols = list(word) + [\"</w>\"]", "symbols = list(word)")]),
    ("2-learn-and-encode", "encoder applies merges in reverse order", [("    for a, b in merges:\n        out = []", "    for a, b in reversed(merges):\n        out = []")]),
    ("2-learn-and-encode", "encoder applies only the first merge that matches", [("        symbols = out\n    return symbols", "        symbols = out\n        if len(out) != len(list(word)) + 1:\n            break\n    return symbols")]),
    ("2-learn-and-encode", "encoder sorts merges alphabetically", [("    for a, b in merges:\n        out = []", "    for a, b in sorted(merges):\n        out = []")]),
    ("2-learn-and-encode", "learner never stops early", [("        if not pairs:\n            break\n", "        if not pairs:\n            merges.append(('x', 'y'))\n            continue\n")]),
    ("3-compression-vs-vocabulary", "tokens not weighted by count", [("freq * len(encode(word, merges))", "len(encode(word, merges))")]),
    ("3-compression-vs-vocabulary", "vocabulary forgets the merges", [("len(chars) + 1 + len(merges)", "len(chars) + 1")]),
    ("3-compression-vs-vocabulary", "vocabulary forgets </w>", [("len(chars) + 1 + len(merges)", "len(chars) + len(merges)")]),
    ("3-compression-vs-vocabulary", "initial length forgets </w>", [("freq * (len(word) + 1)", "freq * len(word)")]),
    ("3-compression-vs-vocabulary", "ratio inverted", [("initial / n_tokens", "n_tokens / initial")]),
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
    for folder, name, edits in MUTANTS:
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
