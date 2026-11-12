#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/bert/scripts/check_mutants.py
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
    ("1-mask-tokens", "selection probability 0.5 instead of mlm_prob", [("rng.random(ids.shape) < mlm_prob", "rng.random(ids.shape) < 0.5")]),
    ("1-mask-tokens", "always [MASK] (100/0/0)", [("(u < 0.8)", "(u < 1.0)")]),
    ("1-mask-tokens", "no random replacement (80/0/20)", [("(u >= 0.8) & (u < 0.9)", "(u >= 2.0)")]),
    ("1-mask-tokens", "random replacement 20 percent (80/20/0)", [("(u >= 0.8) & (u < 0.9)", "(u >= 0.8)")]),
    ("1-mask-tokens", "special tokens can be selected", [(" & (ids >= n_special)", "")]),
    ("1-mask-tokens", "random ids may be special tokens", [("rng.integers(n_special, vocab_size, ids.shape)", "rng.integers(0, vocab_size, ids.shape)")]),
    ("1-mask-tokens", "labels cover every position", [("np.where(selected, ids, -100)", "ids.copy()")]),
    ("1-mask-tokens", "labels hold the corrupted ids", [("np.where(selected, ids, -100)", "np.where(selected, inputs, -100)")]),
    ("1-mask-tokens", "ids modified in place", [("inputs = ids.copy()", "inputs = ids")]),
    ("2-mlm-loss", "mean over all positions with a dummy label", [("nll = -log_probs[np.arange(z.shape[0]), labels[sel]]\n    return float(nll.mean())", "nll = -log_probs[np.arange(z.shape[0]), labels[sel]]\n    return float(nll.sum() / len(labels))")]),
    ("2-mlm-loss", "sum instead of mean", [("nll.mean()", "nll.sum()")]),
    ("2-mlm-loss", "no max subtraction (overflow)", [("    z = z - z.max(axis=-1, keepdims=True)\n", "")]),
    ("2-mlm-loss", "label 0 treated as ignored", [("sel = labels != ignore_index", "sel = (labels != ignore_index) & (labels != 0)")]),
    ("2-mlm-loss", "softmax over the position axis", [("z - np.log(np.exp(z).sum(axis=-1, keepdims=True))", "z - np.log(np.exp(z).sum(axis=0, keepdims=True))")]),
    ("2-mlm-loss", "empty selection divides by zero", [("    if not sel.any():\n        return 0.0\n", "")]),
    ("3-attention-masks", "causal flag ignored", [("    if causal:\n", "    if False:\n")]),
    ("3-attention-masks", "mask keeps the future (triu)", [("np.tril(np.ones(scores.shape))", "np.triu(np.ones(scores.shape))")]),
    ("3-attention-masks", "no sqrt(d) scaling", [("q @ k.T / np.sqrt(q.shape[-1])", "q @ k.T")]),
    ("3-attention-masks", "mask applied after the softmax", [("        scores = np.where(keep, scores, -1e9)\n", ""), ("    return w @ v, w", "    if causal:\n        w = w * np.tril(np.ones(w.shape))\n    return w @ v, w")]),
    ("3-attention-masks", "zero instead of a large negative", [("np.where(keep, scores, -1e9)", "np.where(keep, scores, 0.0)")]),
    ("3-attention-masks", "softmax over the wrong axis", [("w / w.sum(axis=-1, keepdims=True)", "w / w.sum(axis=0, keepdims=True)")]),
    ("3-attention-masks", "scale by sqrt(d_model) instead of sqrt(d)", [("np.sqrt(q.shape[-1])", "np.sqrt(x.shape[-1])")]),
    ("4-input-embeddings", "missing final [SEP]", [("+ list(b) + [sep_id]\n    segment_ids = [0] * (len(a) + 2) + [1] * (len(b) + 1)", "+ list(b)\n    segment_ids = [0] * (len(a) + 2) + [1] * len(b)")]),
    ("4-input-embeddings", "first [SEP] in segment 1", [("[0] * (len(a) + 2) + [1] * (len(b) + 1)", "[0] * (len(a) + 1) + [1] * (len(b) + 2)")]),
    ("4-input-embeddings", "[CLS] in segment 1", [("[0] * (len(a) + 2) + [1] * (len(b) + 1)", "[1] + [0] * (len(a) + 1) + [1] * (len(b) + 1)")]),
    ("4-input-embeddings", "no position embedding", [(" + pos_table[np.arange(T)]", "")]),
    ("4-input-embeddings", "no segment embedding", [(" + seg_table[segment_ids]", "")]),
    ("4-input-embeddings", "position looked up by token id", [("pos_table[np.arange(T)]", "pos_table[ids]")]),
    ("4-input-embeddings", "sentences in the wrong order", [("[cls_id] + list(a) + [sep_id] + list(b)", "[cls_id] + list(b) + [sep_id] + list(a)")]),
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
