#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/deepseek-v32/scripts/check_mutants.py
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
    ("1-lightning-indexer", "no ReLU", [("np.maximum(dots, 0.0)", "dots")]),
    ("1-lightning-indexer", "ReLU applied after the head sum", [("np.einsum(\"tj,tjs->ts\", w, np.maximum(dots, 0.0))", "np.maximum(np.einsum(\"tj,tjs->ts\", w, dots), 0.0)")]),
    ("1-lightning-indexer", "head weights ignored", [("np.einsum(\"tj,tjs->ts\", w, np.maximum(dots, 0.0))", "np.maximum(dots, 0.0).sum(axis=1)")]),
    ("1-lightning-indexer", "no causal mask", [("return np.where(np.tril(np.ones((L, L), dtype=bool)), raw, -np.inf)", "return raw")]),
    ("1-lightning-indexer", "mean over heads instead of weighted sum", [("np.einsum(\"tj,tjs->ts\", w, np.maximum(dots, 0.0))", "np.maximum(dots, 0.0).mean(axis=1)")]),
    ("2-topk-selection", "bottom-k instead of top-k", [("np.argsort(-I[t])[:n]", "np.argsort(I[t])[:n]")]),
    ("2-topk-selection", "always k keys (ignores short prefixes)", [("n = min(k, t + 1)", "n = k")]),
    ("2-topk-selection", "off by one in k", [("n = min(k, t + 1)", "n = min(k + 1, t + 1)")]),
    ("2-topk-selection", "threshold at zero", [("mask[t, np.argsort(-I[t])[:n]] = True", "mask[t, np.argsort(-I[t])[:n]] = I[t, np.argsort(-I[t])[:n]] > 0")]),
    ("3-sparse-attention", "mask ignored (dense attention)", [("logits = np.where(mask[None], logits, -np.inf)", "logits = np.where(np.tril(np.ones_like(mask))[None], logits, -np.inf)")]),
    ("3-sparse-attention", "bottom-k keys selected", [("mask[t, np.argsort(-I[t])[:n]] = True", "mask[t, np.argsort(I[t])[:n]] = True")]),
    ("3-sparse-attention", "scale omitted", [("/ np.sqrt(d)", "")]),
    ("3-sparse-attention", "softmax over the head axis", [("logits.max(axis=-1, keepdims=True))", "logits.max(axis=0, keepdims=True))"), ("e.sum(axis=-1, keepdims=True)", "e.sum(axis=0, keepdims=True)")]),
    ("3-sparse-attention", "masked logits filled with zero", [("logits = np.where(mask[None], logits, -np.inf)", "logits = np.where(mask[None], logits, 0.0)")]),
    ("4-indexer-kl", "heads not summed (first head only)", [("p = attn.sum(axis=0)", "p = attn[0]")]),
    ("4-indexer-kl", "KL direction reversed", [("np.sum(pS[pos] * (np.log(pS[pos]) - np.log(qS[pos])))", "np.sum(qS[pos] * (np.log(qS[pos]) - np.log(pS[pos])))")]),
    ("4-indexer-kl", "sparse stage not renormalized over the set", [("pS = p[t, S] / p[t, S].sum()", "pS = p[t, S]"), ("qS = softmax(I[t, S])", "qS = softmax(I[t])[S]")]),
    ("4-indexer-kl", "sparse stage keeps the bottom-k keys", [("return np.argsort(-I_row)[:k]", "return np.argsort(I_row)[:k]")]),
    ("4-indexer-kl", "log-ratio uses the wrong sign", [("(np.log(pS[pos]) - np.log(qS[pos]))", "(np.log(qS[pos]) - np.log(pS[pos]))")]),
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
