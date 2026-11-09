#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/clip/scripts/check_mutants.py
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
    # Exercise 1: the loss.
    ("1-contrastive-loss", "features not normalized", [
        ("I_e = l2_normalize(I_f, axis=1)", "I_e = I_f"), ("T_e = l2_normalize(T_f, axis=1)", "T_e = T_f")]),
    ("1-contrastive-loss", "only the image-to-text direction", [
        ("return float((loss_i + loss_t) / 2)", "return float(loss_i)")]),
    ("1-contrastive-loss", "both terms use the same axis", [
        ("_log_softmax(logits, axis=0)", "_log_softmax(logits, axis=1)")]),
    ("1-contrastive-loss", "t used as the scale instead of exp(t)", [
        ("logits = np.exp(t) * (I_e @ T_e.T)", "logits = t * (I_e @ T_e.T)")]),
    ("1-contrastive-loss", "sum over the batch instead of the mean", [
        ("-np.mean(np.diag(_log_softmax(logits, axis=1)))", "-np.sum(np.diag(_log_softmax(logits, axis=1)))"),
        ("-np.mean(np.diag(_log_softmax(logits, axis=0)))", "-np.sum(np.diag(_log_softmax(logits, axis=0)))")]),
    ("1-contrastive-loss", "log-softmax without the max subtraction", [
        ("shifted = x - np.max(x, axis=axis, keepdims=True)", "shifted = x")]),
    ("1-contrastive-loss", "l2_normalize ignores its axis", [
        ("np.linalg.norm(x, axis=axis, keepdims=True)", "np.linalg.norm(x, axis=-1, keepdims=True)")]),
    # Exercise 2: the zero-shot classifier.
    ("2-zero-shot-classifier", "average before normalizing each prompt", [
        ("return _unit(_unit(prompt_embs).mean(axis=0))", "return _unit(prompt_embs.mean(axis=0))")]),
    ("2-zero-shot-classifier", "no renormalization after the average", [
        ("return _unit(_unit(prompt_embs).mean(axis=0))", "return _unit(prompt_embs).mean(axis=0)")]),
    ("2-zero-shot-classifier", "average over the class axis", [
        ("_unit(prompt_embs).mean(axis=0)", "_unit(prompt_embs).mean(axis=1)")]),
    ("2-zero-shot-classifier", "softmax over images instead of classes", [
        ("logits.max(axis=1, keepdims=True)", "logits.max(axis=0, keepdims=True)"),
        ("return e / e.sum(axis=1, keepdims=True)", "return e / e.sum(axis=0, keepdims=True)")]),
    ("2-zero-shot-classifier", "image embeddings not normalized", [
        ("_unit(image_embs) @ classifier.T", "image_embs @ classifier.T")]),
    ("2-zero-shot-classifier", "t used as the scale instead of exp(t)", [
        ("logits = np.exp(t) *", "logits = t *")]),
    # Exercise 3: the hand-derived gradients.
    ("3-align-toy-modalities", "no gradient through the normalization", [
        ("dIf = (dIe - Ie * np.sum(Ie * dIe, axis=1, keepdims=True)) / nI", "dIf = dIe"),
        ("dTf = (dTe - Te * np.sum(Te * dTe, axis=1, keepdims=True)) / nT", "dTf = dTe")]),
    ("3-align-toy-modalities", "text gradient forgets the transpose of G", [
        ("dTe = scale * (G.T @ Ie)", "dTe = scale * (G @ Ie)")]),
    ("3-align-toy-modalities", "gradient of the image-to-text term only", [
        ("G = (P_rows + P_cols - 2 * np.eye(n)) / (2 * n)", "G = (P_rows - np.eye(n)) / n")]),
    ("3-align-toy-modalities", "missing the 1/2 of the average", [
        ("G = (P_rows + P_cols - 2 * np.eye(n)) / (2 * n)", "G = (P_rows + P_cols - 2 * np.eye(n)) / n")]),
    ("3-align-toy-modalities", "temperature gradient forgets the scale", [
        ("dt = np.sum(G * L)", "dt = np.sum(G * (Ie @ Te.T))")]),
    ("3-align-toy-modalities", "normalization divides by the squared length", [
        ("dIf = (dIe - Ie * np.sum(Ie * dIe, axis=1, keepdims=True)) / nI", "dIf = (dIe - Ie * np.sum(Ie * dIe, axis=1, keepdims=True)) / nI**2")]),
]


def run_tests(ex, source):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "submission.py").write_text(source)
        shutil.copy(ex / "tests.py", tmp / "tests.py")
        shutil.copy(HARNESS, tmp / "harness.py")
        proc = subprocess.run([sys.executable, "harness.py", "."], cwd=tmp, capture_output=True, text=True, timeout=60)
        lines = proc.stdout.strip().splitlines()
        if not lines:
            return None, proc.stderr[-400:]
        return json.loads(lines[-1]), ""


def main():
    survivors = 0
    for folder, name, edits in MUTANTS:
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
    print(f"\n{len(MUTANTS) - survivors} of {len(MUTANTS)} mutants caught, {survivors} survived")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
