#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/wav2vec2/scripts/check_mutants.py
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
    ("1-encoder-downsampling", "frames from one division by the total stride", [("        n = (n - k) // s + 1\n    return n", "        pass\n    return n // 320")]),
    ("1-encoder-downsampling", "ceil instead of floor", [("n = (n - k) // s + 1\n    return n", "n = -(-(n - k) // s) + 1\n    return n")]),
    ("1-encoder-downsampling", "receptive field adds (k - 1) without the jump", [("rf += (k - 1) * jump", "rf += (k - 1)")]),
    ("1-encoder-downsampling", "jump updated before the receptive-field term", [("        rf += (k - 1) * jump\n        jump *= s", "        jump *= s\n        rf += (k - 1) * jump")]),
    ("1-encoder-downsampling", "short input does not give 0 frames", [("        if n < k:\n            return 0\n", "")]),
    ("2-span-masking", "only the start index is masked", [("mask[s:s + M] = True", "mask[s] = True")]),
    ("2-span-masking", "span is off by one (M - 1 steps)", [("mask[s:s + M] = True", "mask[s:s + M - 1] = True")]),
    ("2-span-masking", "overlap toggles instead of unions", [("mask[s:s + M] = True", "mask[s:s + M] = ~mask[s:s + M]")]),
    ("2-span-masking", "starts sampled with replacement", [("replace=False", "replace=True")]),
    ("2-span-masking", "p read as the masked fraction, so p * T / M starts", [("    n = int(round(p * T))", "    n = int(round(p * T / 10))")]),
    ("3-contrastive-loss", "dot product instead of cosine", [("cand = qn[[t, *distractors[i]]]\n        logits = cand @ cn[t] / kappa", "cand = q[[t, *distractors[i]]]\n        logits = cand @ c[t] / kappa")]),
    ("3-contrastive-loss", "temperature multiplied instead of divided", [("cand @ cn[t] / kappa", "cand @ cn[t] * kappa")]),
    ("3-contrastive-loss", "positive left out of the denominator", [("lse = m + np.log(np.exp(logits - m).sum())", "lse = np.log(np.exp(logits[1:] - m).sum()) + m")]),
    ("3-contrastive-loss", "distractors may include the positive step", [("others = masked_idx[masked_idx != t]", "others = masked_idx")]),
    ("3-contrastive-loss", "distractors drawn from all steps, not only masked ones", [("others = masked_idx[masked_idx != t]", "others = np.arange(int(masked_idx.max()) + 1)\n        others = others[others != t]")]),
    ("3-contrastive-loss", "sum over masked steps instead of the mean", [("return float(total / len(masked_idx))", "return float(total)")]),
    ("3-contrastive-loss", "unstable softmax (no max subtraction)", [("m = logits.max()", "m = 0.0")]),
    ("4-diversity-loss", "average of entropies instead of entropy of the average", [("pbar = p.mean(axis=0)\n    plogp = np.where(pbar > 0, pbar * np.log(np.where(pbar > 0, pbar, 1.0)), 0.0)\n    return float(plogp.sum() / (G * V))", "plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)\n    return float(plogp.sum(axis=(1, 2)).mean() / (G * V))")]),
    ("4-diversity-loss", "normalised by V only, not G * V", [("/ (G * V)", "/ V")]),
    ("4-diversity-loss", "sign flipped (entropy instead of negative entropy)", [("return float(plogp.sum() / (G * V))", "return float(-plogp.sum() / (G * V))")]),
    ("4-diversity-loss", "softmax over the wrong axis", [("axis=-1, keepdims=True)\n    e = np.exp(z)\n    p = e / e.sum(axis=-1, keepdims=True)", "axis=1, keepdims=True)\n    e = np.exp(z)\n    p = e / e.sum(axis=1, keepdims=True)")]),
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
        folder, name, edits = entry
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
