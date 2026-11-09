#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/gpt3/scripts/check_mutants.py
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
    ("1-few-shot-evaluator", "completion is not stripped", [("model(prompt).strip() == answer", "model(prompt) == answer")]),
    ("1-few-shot-evaluator", "substring match instead of exact match", [("model(prompt).strip() == answer", "answer in model(prompt)")]),
    ("1-few-shot-evaluator", "demonstrations may include the query", [("if pq != question", "if True")]),
    ("1-few-shot-evaluator", "demonstrations drawn with replacement", [("replace=False", "replace=True")]),
    ("1-few-shot-evaluator", "single newline between demonstrations", [('A: {a}\\n\\n"', 'A: {a}\\n"')]),
    ("1-few-shot-evaluator", "space after the final A:", [('f"Q: {query}\\nA:"', 'f"Q: {query}\\nA: "')]),
    ("1-few-shot-evaluator", "description after the demonstrations", [("return head + body + f", "return body + head + f")]),
    ("1-few-shot-evaluator", "a fresh generator per query", [("    rng = np.random.default_rng(seed)\n    hits = 0\n    for question, answer in dataset:\n", "    hits = 0\n    for question, answer in dataset:\n        rng = np.random.default_rng(seed)\n")]),
    ("2-multiple-choice-scoring", "ratio divided by length", [("return float(ctx.sum() - unc.sum())", "return float((ctx.sum() - unc.sum()) / len(ctx))")]),
    ("2-multiple-choice-scoring", "ratio sign inverted", [("return float(ctx.sum() - unc.sum())", "return float(unc.sum() - ctx.sum())")]),
    ("2-multiple-choice-scoring", "mean mode returns the total", [("return float(ctx.mean())", "return float(ctx.sum())")]),
    ("2-multiple-choice-scoring", "pick the lowest score", [("np.argmax(", "np.argmin(")]),
    ("2-multiple-choice-scoring", "ratio ignores the unconditional term", [("return float(ctx.sum() - unc.sum())", "return float(ctx.sum())")]),
    ("3-model-size-and-compute", "untied output projection", [("n_vocab * d_model + n_ctx", "2 * n_vocab * d_model + n_ctx")]),
    ("3-model-size-and-compute", "feed-forward width 2d instead of 4d", [("ffn = d * 4 * d + 4 * d + 4 * d * d + d", "ffn = d * 2 * d + 2 * d + 2 * d * d + d")]),
    ("3-model-size-and-compute", "no biases in attention", [("attn = d * 3 * d + 3 * d + d * d + d", "attn = d * 3 * d + d * d")]),
    ("3-model-size-and-compute", "no position embeddings", [("+ n_ctx * d_model", "")]),
    ("3-model-size-and-compute", "2ND instead of 6ND", [("return 6 * n_params * n_tokens", "return 2 * n_params * n_tokens")]),
    ("3-model-size-and-compute", "PF-day of 8.64e18", [("flops / 8.64e19", "flops / 8.64e18")]),
    ("3-model-size-and-compute", "attention flops relative to weights only", [("return attn / (attn + weights)", "return attn / weights")]),
    ("4-sequence-packing", "no end-of-text after the last document", [("        stream.append(eot)\n", "        if doc is not docs[-1]:\n            stream.append(eot)\n")]),
    ("4-sequence-packing", "short remainder kept", [("n_seq = len(stream) // n_ctx", "n_seq = -(-len(stream) // n_ctx)")]),
    ("4-sequence-packing", "end-of-text starts the next document", [("np.concatenate([[0], np.cumsum(ends)[:-1]])", "np.cumsum(ends)")]),
    ("4-sequence-packing", "full row instead of the causal triangle", [("ids[: i + 1] != ids[i]", "ids != ids[i]")]),
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
