#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/e5-mistral/scripts/check_mutants.py
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
    # Exercise 1: last-token pooling.
    ("1-eos-pooling", "always take position -1", [
        ("idx = T - 1 - np.argmax(mask[:, ::-1], axis=1)", "idx = np.full(B, T - 1)")]),
    ("1-eos-pooling", "sum(mask) - 1 breaks left padding", [
        ("idx = T - 1 - np.argmax(mask[:, ::-1], axis=1)", "idx = mask.sum(axis=1) - 1")]),
    ("1-eos-pooling", "first real token instead of the last", [
        ("idx = T - 1 - np.argmax(mask[:, ::-1], axis=1)", "idx = np.argmax(mask, axis=1)")]),
    ("1-eos-pooling", "mean pooling over real tokens", [
        ("return hidden[np.arange(B), idx]", "return (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)")]),
    ("1-eos-pooling", "embed skips the normalization", [
        ("return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)", "return pooled")]),
    # Exercise 2: InfoNCE.
    ("2-infonce-hard-negatives", "no normalization", [
        ("q = _unit(q)", "q = np.asarray(q, dtype=float)"), ("cand = _unit(d_pos)", "cand = np.asarray(d_pos, dtype=float)"),
        ("cand = np.concatenate([cand, _unit(d_hard)], axis=0)", "cand = np.concatenate([cand, np.asarray(d_hard, dtype=float)], axis=0)")]),
    ("2-infonce-hard-negatives", "tau multiplies instead of divides", [
        ("scores = (q @ cand.T) / tau", "scores = (q @ cand.T) * tau")]),
    ("2-infonce-hard-negatives", "hard negatives ignored", [
        ("if d_hard is not None:", "if False:")]),
    ("2-infonce-hard-negatives", "only the query's own hard negative", [
        ("cand = np.concatenate([cand, _unit(d_hard)], axis=0)\n    scores = (q @ cand.T) / tau",
         "scores = np.concatenate([(q @ cand.T), np.sum(q * _unit(d_hard), axis=1, keepdims=True)], axis=1) / tau")]),
    ("2-infonce-hard-negatives", "no in-batch negatives", [
        ("scores = (q @ cand.T) / tau", "scores = np.stack([np.sum(q * _unit(d_pos), axis=1)] + ([np.sum(q * _unit(d_hard), axis=1)] if d_hard is not None else []), axis=1) / tau"),
        ("pos = scores[np.arange(len(q)), np.arange(len(q))]", "pos = scores[:, 0]")]),
    ("2-infonce-hard-negatives", "sum instead of mean", [
        ("np.mean(log_denom - pos)", "np.sum(log_denom - pos)")]),
    ("2-infonce-hard-negatives", "no max subtraction", [
        ("m = scores.max(axis=1, keepdims=True)", "m = np.zeros((len(scores), 1))")]),
    # Exercise 3: retrieval metrics.
    ("3-retrieval-metrics", "recall divides by k", [
        ("len(set(ranked[:k]) & relevant) / len(relevant)", "len(set(ranked[:k]) & relevant) / k")]),
    ("3-retrieval-metrics", "discount log2(rank) with a 1-based rank offset dropped", [
        ("dcg = sum(1.0 / math.log2(j + 2)", "dcg = sum(1.0 / math.log2(j + 1.0000001)")]),
    ("3-retrieval-metrics", "ideal DCG uses all relevant documents", [
        ("for j in range(min(k, len(relevant))))", "for j in range(len(relevant)))")]),
    ("3-retrieval-metrics", "DCG ignores the cutoff k", [
        ("for j, d in enumerate(ranked[:k]) if d in relevant)", "for j, d in enumerate(ranked) if d in relevant)")]),
    ("3-retrieval-metrics", "recall ignores the cutoff k", [
        ("len(set(ranked[:k]) & relevant)", "len(set(ranked) & relevant)")]),
    ("3-retrieval-metrics", "natural log discount", [
        ("1.0 / math.log2(j + 2) for j, d", "1.0 / math.log(j + 2) for j, d")]),
    # Exercise 4: query instruction.
    ("4-query-instruction", "instruction also on documents", [
        ("list(docs)", "[format_query(task, d) for d in docs]")]),
    ("4-query-instruction", "no newline between the lines", [
        ("\\nQuery:", " Query:")]),
    ("4-query-instruction", "add_eos mutates its input", [
        ("return list(token_ids) + [eos_id]", "token_ids.append(eos_id)\n    return token_ids")]),
    ("4-query-instruction", "docs list is the same object", [
        ("list(docs)", "docs")]),
]


def run_tests(ex, source):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "submission.py").write_text(source)
        shutil.copy(ex / "tests.py", tmp / "tests.py")
        shutil.copy(HARNESS, tmp / "harness.py")
        proc = subprocess.run([sys.executable, "harness.py", "."], cwd=tmp, capture_output=True, text=True, timeout=60)
        lines = proc.stdout.strip().splitlines()
        if not lines:
            return None, proc.stderr[-400:]
        return json.loads(lines[-1]), ""


def main():
    survivors = 0
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
    print(f"\n{len(MUTANTS) - survivors} of {len(MUTANTS)} mutants caught, {survivors} survived")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
