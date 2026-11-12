#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/conformer/scripts/check_mutants.py
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
    ("1-depthwise-conv", "filter is flipped (true convolution)", [("w[i] * xp[i:i + T]", "w[k - 1 - i] * xp[i:i + T]")]),
    ("1-depthwise-conv", "even kernel pads (k // 2) at the start", [("left = (k - 1) // 2", "left = k // 2")]),
    ("1-depthwise-conv", "even kernel pads (k - 1) // 2 on both sides", [("right = k - 1 - left", "right = left")]),
    ("1-depthwise-conv", "channels are mixed by a mean", [("out += w[i] * xp[i:i + T]", "out += w[i] * xp[i:i + T].mean(axis=1, keepdims=True)")]),
    ("1-depthwise-conv", "writes into the input", [("xp = np.pad(x, ((left, right), (0, 0)))\n    out = np.zeros((T, d))", "xp = np.pad(x, ((left, right), (0, 0)))\n    x *= 1.0001\n    out = np.zeros((T, d))")]),
    ("2-conv-module", "GLU gates with the first half", [("g = h[:, :d] * sigmoid(h[:, d:])", "g = h[:, d:] * sigmoid(h[:, :d])")]),
    ("2-conv-module", "GLU without the sigmoid", [("g = h[:, :d] * sigmoid(h[:, d:])", "g = h[:, :d] * h[:, d:]")]),
    ("2-conv-module", "batch norm over channels instead of time", [("z = (z - z.mean(axis=0)) / np.sqrt(z.var(axis=0) + eps) * gamma + beta", "z = (z - z.mean(axis=1, keepdims=True)) / np.sqrt(z.var(axis=1, keepdims=True) + eps) * gamma + beta")]),
    ("2-conv-module", "no Swish", [("    z = z * sigmoid(z)\n", "")]),
    ("2-conv-module", "no layer norm at the start", [("    x = (x - mu) / np.sqrt(var + eps)\n", "")]),
    ("2-conv-module", "epsilon dropped from batch norm", [("np.sqrt(z.var(axis=0) + eps)", "np.sqrt(z.var(axis=0))")]),
    ("2-conv-module", "no gamma and beta", [(" * gamma + beta", "")]),
    ("2-conv-module", "unbiased variance in batch norm", [("z.var(axis=0) + eps", "z.var(axis=0, ddof=1) + eps")]),
    ("3-macaron-block", "full-step first feed-forward residual", [("x + 0.5 * ffn1(x)", "x + ffn1(x)")]),
    ("3-macaron-block", "full-step second feed-forward residual", [("x3 + 0.5 * ffn2(x3)", "x3 + ffn2(x3)")]),
    ("3-macaron-block", "convolution before attention", [("x2 = x1 + mhsa(x1)\n    x3 = x2 + conv(x2)", "x2 = x1 + conv(x1)\n    x3 = x2 + mhsa(x2)")]),
    ("3-macaron-block", "second feed-forward reads the wrong input", [("0.5 * ffn2(x3)", "0.5 * ffn2(x1)")]),
    ("3-macaron-block", "no final layer norm", [("return final_ln(x3 + 0.5 * ffn2(x3))", "return x3 + 0.5 * ffn2(x3)")]),
    ("3-macaron-block", "second feed-forward uses the first module", [("0.5 * ffn2(x3)", "0.5 * ffn1(x3)")]),
    ("4-relative-scores", "offset sign reversed (j - i)", [("idx = (T - 1) - np.arange(T)[:, None] + np.arange(T)[None, :]", "idx = (T - 1) + np.arange(T)[:, None] - np.arange(T)[None, :]")]),
    ("4-relative-scores", "no u term", [("content = (q + u) @ k.T", "content = q @ k.T")]),
    ("4-relative-scores", "no v term", [("pos = (q + v) @ p.T", "pos = q @ p.T")]),
    ("4-relative-scores", "no scaling by sqrt(dk)", [("return (content + pos) / np.sqrt(dk)", "return content + pos")]),
    ("4-relative-scores", "scaling by dk instead of sqrt(dk)", [("/ np.sqrt(dk)", "/ dk")]),
    ("4-relative-scores", "position row index off by one", [("idx = (T - 1) -", "idx = T -")]),
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
