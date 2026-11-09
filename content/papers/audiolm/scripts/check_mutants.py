#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/audiolm/scripts/check_mutants.py
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
    ("1-flatten-acoustic", "column-major flattening", [("flat = Y.reshape(-1)", "flat = Y.T.reshape(-1)")]),
    ("1-flatten-acoustic", "no per-layer offset", [("return flat + (np.arange(flat.size) % Q) * N", "return flat")]),
    ("1-flatten-acoustic", "offset by time step instead of layer", [("(np.arange(flat.size) % Q) * N", "(np.arange(flat.size) // Q) * N")]),
    ("1-flatten-acoustic", "unflatten reshapes to (Q, T_A)", [("ids.reshape(-1, Q) - np.arange(Q) * N", "ids.reshape(Q, -1).T - np.arange(Q) * N")]),
    ("1-flatten-acoustic", "no range check on symbols", [("if Y.min() < 0 or Y.max() >= N:", "if False:")]),
    ("2-sequence-lengths", "acoustic hop 640 instead of 320", [("TA = T // acoustic_hop", "TA = T // semantic_hop")]),
    ("2-sequence-lengths", "stage 2 uses all Q layers", [('"stage2": TS + TA * Qp', '"stage2": TS + TA * Q')]),
    ("2-sequence-lengths", "stage 3 chunk includes semantic tokens", [('"stage3_chunk": chunk_TA * Q', '"stage3_chunk": chunk_TA * Q + chunk_s * sr // semantic_hop')]),
    ("2-sequence-lengths", "stage 3 chunk count rounds down", [("math.ceil(seconds / chunk_s)", "int(seconds // chunk_s)")]),
    ("2-sequence-lengths", "stage 3 length grows with the clip", [('"stage3_chunk": chunk_TA * Q', '"stage3_chunk": TA * Q')]),
    ("3-stage2-sequence", "all Q layers instead of Q'", [("np.asarray(Y)[:, :Qp]", "np.asarray(Y)")]),
    ("3-stage2-sequence", "acoustic ids not shifted by K", [("coarse + np.arange(Qp) * N + K", "coarse + np.arange(Qp) * N")]),
    ("3-stage2-sequence", "no per-layer offset", [("coarse + np.arange(Qp) * N + K", "coarse + K")]),
    ("3-stage2-sequence", "loss on the semantic tokens too", [("np.zeros(len(z), dtype=bool)", "np.ones(len(z), dtype=bool)")]),
    ("3-stage2-sequence", "column-major interleaving", [("(coarse + np.arange(Qp) * N + K).reshape(-1)", "(coarse + np.arange(Qp) * N + K).T.reshape(-1)")]),
    ("4-semantic-tokens", "no feature normalization", [("norm = (np.asarray(feats, dtype=float) - mean) / std", "norm = np.asarray(feats, dtype=float)")]),
    ("4-semantic-tokens", "mean subtracted but not divided by std", [("(np.asarray(feats, dtype=float) - mean) / std", "(np.asarray(feats, dtype=float) - mean)")]),
    ("4-semantic-tokens", "farthest centroid (argmax)", [("d.argmin(axis=1)", "d.argmax(axis=1)")]),
    ("4-semantic-tokens", "collapse removes all duplicates", [("keep = np.concatenate([[True], tokens[1:] != tokens[:-1]])", "keep = np.array([t not in tokens[:i] for i, t in enumerate(tokens)])")]),
    ("4-semantic-tokens", "run lengths off by one at the end", [("np.diff(np.append(starts, len(tokens)))", "np.diff(np.append(starts, len(tokens) - 1))")]),
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
