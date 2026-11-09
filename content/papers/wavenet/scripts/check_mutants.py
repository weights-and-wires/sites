#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/wavenet/scripts/check_mutants.py [--skip-torch]
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
    ("1-causal-dilated-conv", "pad on the right, so outputs see the future", [("np.concatenate([np.zeros(pad), x])", "np.concatenate([x, np.zeros(pad)])")]),
    ("1-causal-dilated-conv", "filter taps in reverse order", [("w[j] * xp[", "w[k - 1 - j] * xp[")]),
    ("1-causal-dilated-conv", "dilation ignored (acts like dilation 1)", [("pad = (k - 1) * dilation", "pad = k - 1"), ("xp[j * dilation : j * dilation + len(x)]", "xp[j : j + len(x)]")]),
    ("2-receptive-field", "formula forgets the (k - 1) factor", [("(kernel_size - 1) * sum(dilations)", "sum(dilations)")]),
    ("2-receptive-field", "formula counts layers + k - 1 (ignores dilation)", [("1 + (kernel_size - 1) * sum(dilations)", "len(dilations) + kernel_size - 1")]),
    ("2-receptive-field", "brute force counts nonzero outputs instead of the span", [("int(np.nonzero(y)[0].max()) + 1", "int(np.count_nonzero(y))")]),
    ("2-receptive-field", "brute force off by one", [("int(np.nonzero(y)[0].max()) + 1", "int(np.nonzero(y)[0].max())")]),
    ("3-mu-law", "truncate instead of round", [("(f + 1) / 2 * mu + 0.5", "(f + 1) / 2 * mu")]),
    ("3-mu-law", "linear quantization (no companding)", [("f = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)", "f = x")]),
    ("3-mu-law", "decode uses log(mu) instead of log(1 + mu)", [("np.abs(f) * np.log1p(mu)", "np.abs(f) * np.log(mu)")]),
    ("3-mu-law", "256 levels indexed with mu + 1", [("(f + 1) / 2 * mu + 0.5", "(f + 1) / 2 * (mu + 1) + 0.5")]),
    ("3-mu-law", "decode drops the sign handling", [("np.sign(f) * np.expm1(np.abs(f)", "np.expm1(np.abs(f)")]),
    ("4-gated-unit", "tanh and sigmoid swapped", [("np.tanh(causal_conv(x, Wf, dilation)) * sigmoid(causal_conv(x, Wg, dilation))", "sigmoid(causal_conv(x, Wf, dilation)) * np.tanh(causal_conv(x, Wg, dilation))")]),
    ("4-gated-unit", "gate reuses the filter weights", [("sigmoid(causal_conv(x, Wg, dilation))", "sigmoid(causal_conv(x, Wf, dilation))")]),
    ("4-gated-unit", "ReLU instead of tanh on the filter", [("np.tanh(causal_conv(x, Wf, dilation))", "np.maximum(causal_conv(x, Wf, dilation), 0)")]),
    ("4-gated-unit", "gate ignores the dilation", [("causal_conv(x, Wg, dilation)", "causal_conv(x, Wg, 1)")]),
    ("4-gated-unit", "sigmoid with the wrong sign", [("1.0 / (1.0 + np.exp(-a))", "1.0 / (1.0 + np.exp(a))")]),
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
