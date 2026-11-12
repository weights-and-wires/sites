#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/ds2/scripts/check_mutants.py
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
    ("1-log-spectrogram", "rectangular window", [("window = np.hanning(win)", "window = np.ones(win)")]),
    ("1-log-spectrogram", "amplitude instead of power", [("np.abs(spec) ** 2", "np.abs(spec)")]),
    ("1-log-spectrogram", "hop equals window (no overlap)", [("x[i * hop : i * hop + win]", "x[i * win : i * win + win]"), ("n_frames = 1 + (len(x) - win) // hop", "n_frames = len(x) // win")]),
    ("1-log-spectrogram", "log10 instead of natural log", [("np.log(power + eps)", "np.log10(power + eps)")]),
    ("1-log-spectrogram", "eps added after the log", [("np.log(power + eps)", "np.log(power) + eps")]),
    ("1-log-spectrogram", "FFT over the frame axis", [("np.fft.rfft(frames, axis=1)", "np.fft.rfft(frames, axis=0)")]),
    ("1-log-spectrogram", "off-by-one frame count", [("n_frames = 1 + (len(x) - win) // hop", "n_frames = (len(x) - win) // hop")]),
    ("1-log-spectrogram", "frames shifted by one sample", [("x[i * hop : i * hop + win]", "x[i * hop + 1 : i * hop + win + 1]")]),
    ("2-sequence-batchnorm", "sequence-wise statistics over the batch only", [("valid = a[mask]  # (N, d)\n    mean = valid.mean(axis=0)\n    var = valid.var(axis=0)", "mean = a.mean(axis=0, keepdims=True)\n    var = a.var(axis=0, keepdims=True)")]),
    ("2-sequence-batchnorm", "padding enters the statistics", [("valid = a[mask]  # (N, d)", "valid = a.reshape(-1, d)")]),
    ("2-sequence-batchnorm", "padded positions not zeroed", [("return np.where(mask[:, :, None], out, 0.0)", "return out")]),
    ("2-sequence-batchnorm", "unbiased variance", [("var = valid.var(axis=0)", "var = valid.var(axis=0, ddof=1)")]),
    ("2-sequence-batchnorm", "epsilon outside the square root", [("np.sqrt(var + eps)", "(np.sqrt(var) + eps)")]),
    ("2-sequence-batchnorm", "one mean and variance for all units", [("mean = valid.mean(axis=0)\n    var = valid.var(axis=0)", "mean = valid.mean()\n    var = valid.var()")]),
    ("2-sequence-batchnorm", "gamma and beta swapped", [("gamma * (a - mean) / np.sqrt(var + eps) + beta\n    return np.where", "beta * (a - mean) / np.sqrt(var + eps) + gamma\n    return np.where")]),
    ("2-sequence-batchnorm", "frame-wise pools time steps", [("a.mean(axis=0, keepdims=True)\n    var = a.var(axis=0, keepdims=True)", "a.mean(axis=(0, 1), keepdims=True)\n    var = a.var(axis=(0, 1), keepdims=True)")]),
    ("2-sequence-batchnorm", "frame-wise normalizes over time instead of batch", [("a.mean(axis=0, keepdims=True)\n    var = a.var(axis=0, keepdims=True)", "a.mean(axis=1, keepdims=True)\n    var = a.var(axis=1, keepdims=True)")]),
    ("3-row-convolution", "looks at the past instead of the future", [("padded = np.vstack([h, np.zeros((tau, d))])", "padded = np.vstack([np.zeros((tau, d)), h])"), ("padded[j : j + T]", "padded[tau - j : tau - j + T]")]),
    ("3-row-convolution", "filter taps in reverse order", [("W[:, j] * padded[j : j + T]", "W[:, tau - j] * padded[j : j + T]")]),
    ("3-row-convolution", "mixes features (matrix product)", [("r += W[:, j] * padded[j : j + T]", "r += padded[j : j + T] @ np.diag(W[:, j]) @ np.ones((d, d)) / d")]),
    ("3-row-convolution", "wraps around instead of zero padding", [("padded = np.vstack([h, np.zeros((tau, d))])", "padded = np.vstack([h, h[:tau]])")]),
    ("3-row-convolution", "repeats the last frame instead of zero padding", [("padded = np.vstack([h, np.zeros((tau, d))])", "padded = np.vstack([h, np.repeat(h[-1:], tau, axis=0)])")]),
    ("3-row-convolution", "drops the current frame (starts at t + 1)", [("padded[j : j + T]", "padded[min(j + 1, tau) : min(j + 1, tau) + T]")]),
    ("3-row-convolution", "writes into the input", [("padded = np.vstack([h, np.zeros((tau, d))])", "padded = np.vstack([h, np.zeros((tau, d))]); h *= 2")]),
    ("4-stride-and-bigrams", "bigrams cross word boundaries", [("for k, word in enumerate(text.split(\" \")):\n        if k > 0:\n            tokens.append(\" \")\n        for i in range(0, len(word), 2):\n            tokens.append(word[i : i + 2])", "tokens = [text[i : i + 2] for i in range(0, len(text), 2)]")]),
    ("4-stride-and-bigrams", "space dropped from the tokens", [("            tokens.append(\" \")\n", "            pass\n")]),
    ("4-stride-and-bigrams", "overlapping bigrams", [("range(0, len(word), 2)", "range(0, max(len(word) - 1, 1))")]),
    ("4-stride-and-bigrams", "floor instead of ceiling", [("T = math.ceil(T / s)", "T = T // s")]),
    ("4-stride-and-bigrams", "forgets blanks between repeats", [("return len(tokens) + repeats", "return len(tokens)")]),
    ("4-stride-and-bigrams", "counts every repeat as two", [("return len(tokens) + repeats", "return len(tokens) + 2 * repeats")]),
    ("4-stride-and-bigrams", "max_stride off by one", [("best = s\n    return best", "best = s + 1\n    return best")]),
    ("4-stride-and-bigrams", "max_stride returns 1 when impossible", [("    best = 0\n", "    best = 1\n")]),
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
