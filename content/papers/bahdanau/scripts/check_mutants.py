#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/bahdanau/scripts/check_mutants.py
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
    ("1-additive-attention", "no tanh (score splits into an s term and an h term)", [("np.tanh(Wa @ s_prev + H @ Ua.T)", "(Wa @ s_prev + H @ Ua.T)")]),
    ("1-additive-attention", "decoder state ignored", [("np.tanh(Wa @ s_prev + H @ Ua.T)", "np.tanh(H @ Ua.T)")]),
    ("1-additive-attention", "weights not normalized (exp only)", [("alpha = softmax(e)", "alpha = np.exp(e)")]),
    ("1-additive-attention", "context is the plain mean of the annotations", [("return alpha @ H, alpha", "return H.mean(axis=0), alpha")]),
    ("1-additive-attention", "context uses the last annotation only (fixed vector)", [("return alpha @ H, alpha", "return H[-1], alpha")]),
    ("2-alignment-matrix", "softmax over decoder steps (axis 0)", [("alpha = softmax(e, axis=1)", "alpha = softmax(e, axis=0)")]),
    ("2-alignment-matrix", "no tanh", [("np.tanh(A[:, None, :] + B[None, :, :])", "(A[:, None, :] + B[None, :, :])")]),
    ("2-alignment-matrix", "tanh applied to each term before the sum", [("np.tanh(A[:, None, :] + B[None, :, :])", "(np.tanh(A)[:, None, :] + np.tanh(B)[None, :, :])")]),
    ("2-alignment-matrix", "all steps share the first decoder state", [("A = S_prev @ Wa.T", "A = np.repeat(S_prev[:1], S_prev.shape[0], axis=0) @ Wa.T")]),
    ("3-bidirectional-annotations", "backward states left in reading order (not re-aligned)", [("bwd[t] = prev", "bwd[Tx - 1 - t] = prev")]),
    ("3-bidirectional-annotations", "backward RNN reuses the forward weights", [("np.tanh(Wb @ X[t] + Ub @ prev)", "np.tanh(Wf @ X[t] + Uf @ prev)")]),
    ("3-bidirectional-annotations", "backward half placed first", [("np.concatenate([fwd, bwd], axis=1)", "np.concatenate([bwd, fwd], axis=1)")]),
    ("3-bidirectional-annotations", "two forward passes (no backward reading)", [("for t in reversed(range(Tx)):", "for t in range(Tx):")]),
    ("4-attention-backward", "missing the alpha . dalpha term", [("dE = alpha * (dalpha - np.sum(alpha * dalpha, axis=1, keepdims=True))", "dE = alpha * dalpha")]),
    ("4-attention-backward", "sum over the wrong axis", [("np.sum(alpha * dalpha, axis=1, keepdims=True)", "np.sum(alpha * dalpha, axis=0, keepdims=True)")]),
    ("4-attention-backward", "dalpha built from G alone, annotations ignored", [("dalpha = G @ H.T", "dalpha = G @ G.T @ alpha")]),
    ("4-attention-backward", "dH ignores the weights (every annotation gets the summed gradient)", [("dH = alpha.T @ G", "dH = np.ones((H.shape[0], 1)) * G.sum(axis=0)")]),
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
