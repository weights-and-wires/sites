#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/transformer/scripts/check_mutants.py [--skip-torch]
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
# Known gap: exercise 5 can't catch a softmax over the query axis, because the
# model still learns modular addition with it. The NumPy exercises cover that bug.
MUTANTS = [
    ("1-attention", "softmax over the wrong axis (-2)", [("weights = softmax(scores, axis=-1)", "weights = softmax(scores, axis=-2)")]),
    ("1-attention", "softmax ignores its axis argument", [("axis=axis", "axis=-1")]),
    ("1-attention", "softmax without the max subtraction", [("shifted = x - np.max(x, axis=axis, keepdims=True)", "shifted = x")]),
    ("1-attention", "mask with 0 instead of -1e9", [("np.where(mask, scores, -1e9)", "np.where(mask, scores, 0.0)")]),
    ("1-attention", "mask polarity inverted", [("np.where(mask, scores, -1e9)", "np.where(~mask, scores, -1e9)")]),
    ("1-attention", "divide by sqrt(d_v) instead of sqrt(d_k)", [("/ np.sqrt(d_k)", "/ np.sqrt(V.shape[-1])")]),
    ("1-attention", "no scaling at all", [("/ np.sqrt(d_k)", "")]),
    ("1-attention", "K.T instead of swapaxes (breaks batches)", [("np.swapaxes(K, -1, -2)", "K.T")]),
    ("2-causal-mask", "upper triangle instead of lower", [("np.tril(np.ones((n, n), dtype=bool))", "np.triu(np.ones((n, n), dtype=bool))")]),
    ("2-causal-mask", "strict lower triangle (diagonal hidden)", [("np.tril(np.ones((n, n), dtype=bool))", "np.tril(np.ones((n, n), dtype=bool), k=-1)")]),
    ("2-causal-mask", "float mask instead of boolean", [("np.tril(np.ones((n, n), dtype=bool))", "np.tril(np.ones((n, n)))")]),
    ("2-causal-mask", "all True (no masking)", [("np.tril(np.ones((n, n), dtype=bool))", "np.ones((n, n), dtype=bool)")]),
    ("3-positional-encoding", "sin and cos swapped", [("pe[:, 0::2] = np.sin(angle)", "pe[:, 0::2] = np.cos(angle)"), ("pe[:, 1::2] = np.cos(angle)", "pe[:, 1::2] = np.sin(angle)")]),
    ("3-positional-encoding", "halves concatenated instead of interleaved", [("pe[:, 0::2] = np.sin(angle)", "pe[:, : d // 2] = np.sin(angle)"), ("pe[:, 1::2] = np.cos(angle)", "pe[:, d // 2 :] = np.cos(angle)")]),
    ("3-positional-encoding", "exponent i/d instead of 2i/d", [("10000 ** (2 * i / d)", "10000 ** (i / d)")]),
    ("3-positional-encoding", "base 1000 instead of 10000", [("10000 ** (2 * i / d)", "1000 ** (2 * i / d)")]),
    ("4-multi-head-attention", "reshape to (h, n, d_k) without transpose", [("M.reshape(n, h, d_k).transpose(1, 0, 2)", "M.reshape(h, n, d_k)")]),
    ("4-multi-head-attention", "merge without transposing back", [("heads.transpose(1, 0, 2).reshape(n, h * d_k)", "heads.reshape(n, h * d_k)")]),
    ("4-multi-head-attention", "forget W^O", [("return concat @ Wo", "return concat")]),
    ("4-multi-head-attention", "mask not passed to attention", [("split(V), mask)", "split(V))")]),
    ("4-multi-head-attention", "keys built from queries", [("split(K)", "split(Q)")]),
    ("5-modular-addition", "post-LN instead of pre-LN", [
        ("x = x + self.attn(self.ln1(x))\n        return x + self.ff2(F.relu(self.ff1(self.ln2(x))))",
         "x = self.ln1(x + self.attn(x))\n        return self.ln2(x + self.ff2(F.relu(self.ff1(x))))")], "torch"),
    ("5-modular-addition", "no residual around the FFN", [("return x + self.ff2(", "return self.ff2(")], "torch"),
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
