#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/deepseek-v3/scripts/check_mutants.py
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
    ("1-aux-free-balancing", "gates use a softmax over the chosen affinities, not Equation 13", [("chosen = np.take_along_axis(s, idx, axis=1)", "chosen = np.exp(np.take_along_axis(s, idx, axis=1))")]),
    ("1-aux-free-balancing", "bias ignored when choosing experts", [("np.argsort(-(s + b), axis=1)", "np.argsort(-s, axis=1)")]),
    ("1-aux-free-balancing", "bias update has the wrong sign", [("return b - gamma * np.sign", "return b + gamma * np.sign")]),
    ("1-aux-free-balancing", "bias step proportional to the load gap", [("return b - gamma * np.sign(load - load.mean())", "return b - gamma * (load - load.mean()) / load.mean()")]),
    ("1-aux-free-balancing", "gates not normalized", [("return chosen / chosen.sum(axis=1, keepdims=True)", "return chosen")]),
    ("1-aux-free-balancing", "load counts omit the minlength", [("minlength=n", "minlength=0")]),
    ("2-fine-grained-fp8", "tiles run down columns (128x1) instead of along rows", [("g = x.reshape(r, c // n_c, n_c)\n    return _scale_from_amax(np.abs(g).max(axis=2))", "g = x.T.reshape(c, r // n_c, n_c)\n    return _scale_from_amax(np.abs(g).max(axis=2)).T")]),
    ("2-fine-grained-fp8", "tile roundtrip forgets to unscale", [("return (quantize(g * s) / s).reshape(r, c)\n\n\ndef block", "return quantize(g * s).reshape(r, c)\n\n\ndef block")]),
    ("2-fine-grained-fp8", "scale from the signed maximum", [("np.abs(g).max(axis=2, keepdims=True)", "g.max(axis=2, keepdims=True)")]),
    ("2-fine-grained-fp8", "all-zero tile gets a huge scale", [("amax = np.where(amax == 0, E4M3_MAX, amax)", "amax = np.where(amax == 0, 1e-300, amax)")]),
    ("2-fine-grained-fp8", "block roundtrip uses one scale per tensor", [("np.abs(g).max(axis=(1, 3), keepdims=True)", "np.abs(g).max(keepdims=True)")]),
    ("2-fine-grained-fp8", "tile scale uses the mean, not the max", [("np.abs(g).max(axis=2, keepdims=True)", "np.abs(g).mean(axis=2, keepdims=True) * 4")]),
    ("3-mtp-loss", "targets are off by one (t_{i+k} instead of t_{i+k+1})", [("tokens[k + 1 : k + 1 + logits.shape[0]]", "tokens[k : k + logits.shape[0]]")]),
    ("3-mtp-loss", "loss divided by the number of positions, not T", [("/ T", "/ logits.shape[0]")]),
    ("3-mtp-loss", "RMSNorm skipped before the concatenation", [("z = np.concatenate([rmsnorm(h_prev), rmsnorm(emb_next)], axis=-1)", "z = np.concatenate([np.asarray(h_prev, float), np.asarray(emb_next, float)], axis=-1)")]),
    ("3-mtp-loss", "embedding first in the concatenation", [("np.concatenate([rmsnorm(h_prev), rmsnorm(emb_next)], axis=-1)", "np.concatenate([rmsnorm(emb_next), rmsnorm(h_prev)], axis=-1)")]),
    ("3-mtp-loss", "log-softmax without the max shift", [("z = logits - logits.max(axis=1, keepdims=True)", "z = logits")]),
    ("3-mtp-loss", "total loss sums depths instead of averaging", [("return lam * float(np.mean(losses))", "return lam * float(np.sum(losses))")]),
    ("4-training-cost", "days forget that hours are in thousands", [("return k_per_trillion * 1000 / gpus / 24", "return k_per_trillion / gpus / 24")]),
    ("4-training-cost", "cost forgets that hours are in thousands", [("return k_hours * 1000 * price / 1e6", "return k_hours * price / 1e6")]),
    ("4-training-cost", "pre-training hours use the token count in billions", [("return k_per_trillion * tokens_trillion", "return k_per_trillion * tokens_trillion * 1000")]),
    ("4-training-cost", "shares divide by the largest stage", [("total = sum(stages)", "total = max(stages)")]),
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
