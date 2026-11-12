#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/llada/scripts/check_mutants.py
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
    ("1-forward-masking", "masks with probability 1 - t", [("< t", "> t")]),
    ("1-forward-masking", "one mask row shared by every sequence", [("rng.random(np.shape(x0)) < t", "rng.random(np.shape(x0)[-1]) < t")]),
    ("1-forward-masking", "modifies x0 in place", [("x_t = np.where(masked, mask_id, x0)", "x_t = x0\n    x_t[masked] = mask_id")]),
    ("1-forward-masking", "mask rate t squared", [("< t", "< t * t")]),
    ("2-masked-loss", "no 1/t weight", [("nll[masked].sum() / t", "nll[masked].sum()")]),
    ("2-masked-loss", "weight t instead of 1/t", [("nll[masked].sum() / t", "nll[masked].sum() * t")]),
    ("2-masked-loss", "mean over masked positions", [("nll[masked].sum() / t", "nll[masked].mean() / t")]),
    ("2-masked-loss", "unmasked positions contribute", [("nll[masked].sum() / t", "nll.sum() / t")]),
    ("2-masked-loss", "mean over all positions", [("nll[masked].sum() / t", "nll[masked].sum() / t / len(x0)")]),
    ("2-masked-loss", "log-softmax without the max shift", [("z = logits - logits.max(axis=-1, keepdims=True)", "z = logits")]),
    ("3-remasking-step", "confidence is the raw max logit", [("conf = softmax(logits).max(axis=-1)", "conf = logits.max(axis=-1)")]),
    ("3-remasking-step", "keeps the least confident", [("np.argsort(-conf, kind=\"stable\")", "np.argsort(conf, kind=\"stable\")")]),
    ("3-remasking-step", "visible tokens can be remasked", [("conf = np.where(masked, conf, np.inf)", "conf = np.where(masked, conf, 0.0)")]),
    ("3-remasking-step", "visible tokens overwritten by predictions", [("filled = np.where(masked, pred, x_t)", "filled = pred")]),
    ("3-remasking-step", "ties go to the higher index", [("np.argsort(-conf, kind=\"stable\")[:n_un]", "np.argsort(-conf[::-1], kind=\"stable\")[::-1][:n_un]")]),
    ("3-remasking-step", "random step prefers confident positions", [("rng.permutation(masked_idx)[:n_new]", "masked_idx[np.argsort(-logits.max(axis=-1)[masked_idx])][:n_new]")]),
    ("3-remasking-step", "random step ignores already unmasked count", [("n_new = n_un - (len(x_t) - len(masked_idx))", "n_new = n_un")]),
    ("3-remasking-step", "random step modifies x_t in place", [("out = x_t.copy()", "out = x_t")]),
    ("4-sampler", "float schedule drops a token", [("n_un = length * k // steps", "n_un = int(length * (1 - (1 - k / steps)) - 1e-9)")]),
    ("4-sampler", "ceiling schedule", [("n_un = length * k // steps", "n_un = -(-length * k // steps)")]),
    ("4-sampler", "model called once, stale logits reused", [("        logits = predict(x)\n", "        logits = predict(x) if k == 1 else logits\n")]),
    ("4-sampler", "starts from step 0", [("range(1, steps + 1)", "range(0, steps)")]),
    ("4-sampler", "always low-confidence", [("if remask == \"low_confidence\":\n            x = lowconf_step", "if True:\n            x = lowconf_step")]),
    ("4-sampler", "unknown strategy accepted", [("if remask not in (\"low_confidence\", \"random\"):", "if False:")]),
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
