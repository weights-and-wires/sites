#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/unet/scripts/check_mutants.py 
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
    ("1-valid-sizes", "odd sizes allowed before the pool", [("if s <= 0 or s % 2:", "if s <= 0:")]),
    ("1-valid-sizes", "padded convolutions (subtract 2, not 4)", [("s -= 4", "s -= 2")]),
    ("1-valid-sizes", "up-convolution does not double the size", [("s *= 2", "s *= 1")]),
    ("1-valid-sizes", "bottom convolutions forgotten", [("    s -= 4\n    if s <= 0:\n        return None\n    for _ in range(levels):\n        s *= 2", "    for _ in range(levels):\n        s *= 2")]),
    ("1-valid-sizes", "no check for non-positive sizes", [("if s <= 0 or s % 2:", "if s % 2:"), ("    if s <= 0:\n        return None\n    for", "    for"), ("        if s <= 0:\n            return None\n    return s", "    return s")]),
    ("2-crop-concat", "crop from the top-left corner", [("top = (H - h) // 2", "top = 0"), ("left = (W - w) // 2", "left = 0")]),
    ("2-crop-concat", "up-convolution channels first", [("np.concatenate([cropped, up], axis=0)", "np.concatenate([up, cropped], axis=0)")]),
    ("2-crop-concat", "concatenate on the width axis", [("np.concatenate([cropped, up], axis=0)", "np.concatenate([cropped, up], axis=2)")]),
    ("2-crop-concat", "crop rounds the wrong way", [("top = (H - h) // 2", "top = (H - h + 1) // 2"), ("left = (W - w) // 2", "left = (W - w + 1) // 2")]),
    ("3-weighted-loss", "squares d1 and d2 separately", [("-((d1 + d2) ** 2)", "-(d1**2 + d2**2)")]),
    ("3-weighted-loss", "denominator sigma instead of 2 sigma^2", [("(2 * sigma**2)", "sigma")]),
    ("3-weighted-loss", "drops the minus sign of the loss", [("return float(-(w * true_log_p).sum())", "return float((w * true_log_p).sum())")]),
    ("3-weighted-loss", "ignores the weight map", [("(w * true_log_p).sum()", "true_log_p.sum()")]),
    ("3-weighted-loss", "softmax over the height axis", [("logits.max(axis=0, keepdims=True)", "logits.max(axis=1, keepdims=True)"), ("np.exp(shifted).sum(axis=0, keepdims=True)", "np.exp(shifted).sum(axis=1, keepdims=True)")]),
    ("3-weighted-loss", "softmax without the max subtraction", [("shifted = logits - logits.max(axis=0, keepdims=True)", "shifted = logits")]),
    ("4-tiny-unet", "up-convolution kernel transposed", [("W[:, :, a, c]", "W[:, :, c, a]")]),
    ("4-tiny-unet", "up-convolution ignores its bias", [("return out + b[:, None, None]\n\n\ndef tiny_unet", "return out\n\n\ndef tiny_unet")]),
    ("4-tiny-unet", "skip connection replaced by zeros", [("m = crop_and_concat(d2, u)", "m = crop_and_concat(np.zeros_like(d2), u)")]),
    ("4-tiny-unet", "skip taken before the second convolution", [("m = crop_and_concat(d2, u)", "m = crop_and_concat(d1, u)")]),
    ("4-tiny-unet", "ReLU after the up-convolution", [("u = upconv2x2(b2, *P[\"up\"])", "u = relu(upconv2x2(b2, *P[\"up\"]))")]),
    ("4-tiny-unet", "ReLU after the 1x1 head", [("return conv2d_valid(e2, *P[\"head\"])", "return relu(conv2d_valid(e2, *P[\"head\"]))")]),
    ("4-tiny-unet", "skip cropped from the top-left corner", [("m = crop_and_concat(d2, u)", "m = np.concatenate([d2[:, :u.shape[1], :u.shape[2]], u], axis=0)")]),
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
