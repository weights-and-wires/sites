#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/muon/scripts/check_mutants.py
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
NS_NORM = "X = G / (np.linalg.norm(G) + eps)"
MUTANTS = [
    ("1-newton-schulz", "no normalization before the iteration", [(NS_NORM, "X = G.astype(float)")]),
    ("1-newton-schulz", "normalize by the largest entry, not the Frobenius norm", [(NS_NORM, "X = G / (np.abs(G).max() + eps)")]),
    ("1-newton-schulz", "b and c swapped", [("B = b * A + c * (A @ A)", "B = c * A + b * (A @ A)")]),
    ("1-newton-schulz", "one step too few", [("for _ in range(steps):", "for _ in range(max(steps - 1, 0)):")]),
    ("1-newton-schulz", "baseline coefficients ignore the coeffs argument", [("a, b, c = coeffs", "a, b, c = (3.4445, -4.7750, 2.0315)")]),
    ("1-newton-schulz", "no transpose back for tall input", [("    if tall:\n        X = X.T\n    return X", "    return X")]),
    ("1-newton-schulz", "A = X^T X instead of X X^T", [("A = X @ X.T", "A = X.T @ X")]),
    ("1-newton-schulz", "modifies the input in place", [(NS_NORM, "X = G\n    X /= (np.linalg.norm(G) + eps)")]),
    ("2-muon-step", "momentum as an exponential moving average", [("B_new = mu * B + G", "B_new = mu * B + (1 - mu) * G")]),
    ("2-muon-step", "orthogonalize the gradient, then apply momentum", [
        ("    B_new = mu * B + G\n    O = newton_schulz(B_new)\n    return W - lr * O, B_new",
         "    O = newton_schulz(G)\n    B_new = mu * B + O\n    return W - lr * B_new, B_new")]),
    ("2-muon-step", "step sign flipped", [("return W - lr * O, B_new", "return W + lr * O, B_new")]),
    ("2-muon-step", "forgets to orthogonalize", [("O = newton_schulz(B_new)", "O = B_new")]),
    ("2-muon-step", "orthogonalizes only the new gradient", [("O = newton_schulz(B_new)", "O = newton_schulz(G)")]),
    ("2-muon-step", "momentum buffer is never accumulated", [("B_new = mu * B + G", "B_new = G")]),
    ("3-route-parameters", "substring match for the head", [("name.startswith(ADAMW_PREFIXES)", "any(p in name for p in ADAMW_PREFIXES)")]),
    ("3-route-parameters", "embeddings go to Muon because they are 2D", [('    if name.startswith(ADAMW_PREFIXES):\n        return "adamw"\n', "")]),
    ("3-route-parameters", "vectors go to Muon", [("if len(shape) in (2, 4):", "if len(shape) >= 1:")]),
    ("3-route-parameters", "conv filters go to AdamW", [("if len(shape) in (2, 4):", "if len(shape) == 2:")]),
    ("3-route-parameters", "conv flattened over the first two dimensions", [("G.reshape(G.shape[0], -1)", "G.reshape(-1, G.shape[-1] * G.shape[-2])")]),
    ("4-hybrid-training", "Muon also applied to the input and output matrices", [
        ('for name in ("W_in", "W_out"):\n            new[name] = adamw_update(params[name], grads[name], states[name], lr_adam)',
         'for name in ("W_in", "W_out"):\n            new[name] = params[name] - lr_muon * newton_schulz(grads[name])')]),
    ("4-hybrid-training", "hidden matrix always uses AdamW", [("if use_muon:", "if False:")]),
    ("4-hybrid-training", "momentum buffer reset every step", [("B = mu * B + grads[\"W_h\"]", "B = grads[\"W_h\"]")]),
    ("4-hybrid-training", "Muon step added instead of subtracted", [('params["W_h"] - lr_muon * newton_schulz(B)', 'params["W_h"] + lr_muon * newton_schulz(B)')]),
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
