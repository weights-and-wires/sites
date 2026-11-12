#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/rainbow/scripts/check_mutants.py
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
    ("1-n-step-target", "bootstrap ignored", [("return float(ret + gamma**m * q_boot)", "return float(ret)")]),
    ("1-n-step-target", "bootstrap kept after the episode ends", [("    if done:\n        return float(ret)\n", "")]),
    ("1-n-step-target", "bootstrap discounted by gamma, not gamma^m", [("gamma**m * q_boot", "gamma * q_boot")]),
    ("1-n-step-target", "rewards discounted in reverse order", [("enumerate(rewards)", "enumerate(rewards[::-1])")]),
    ("1-n-step-target", "first reward discounted too", [("gamma**k * r", "gamma**(k + 1) * r")]),
    ("2-c51-projection", "no clipping to the support", [("tz = np.clip(r + gamma_n * z, z[0], z[-1])", "tz = r + gamma_n * z")]),
    ("2-c51-projection", "mass split the wrong way round", [("m[lo[j]] += p[j] * (hi[j] - b[j])\n            m[hi[j]] += p[j] * (b[j] - lo[j])", "m[lo[j]] += p[j] * (b[j] - lo[j])\n            m[hi[j]] += p[j] * (hi[j] - b[j])")]),
    ("2-c51-projection", "mass on an exact atom is dropped", [("            m[lo[j]] += p[j]\n        else", "            pass\n        else")]),
    ("2-c51-projection", "reward not added", [("r + gamma_n * z", "gamma_n * z")]),
    ("2-c51-projection", "discount not applied", [("r + gamma_n * z", "r + z")]),
    ("3-rainbow-target", "target network picks the action (C51)", [("np.asarray(p_online) @ z", "np.asarray(p_target) @ z")]),
    ("3-rainbow-target", "online network also scores the action", [("np.asarray(p_target)[a_star]", "np.asarray(p_online)[a_star]")]),
    ("3-rainbow-target", "discount fixed at gamma, not gamma^m", [("gamma ** len(rewards)", "gamma")]),
    ("3-rainbow-target", "terminal window still discounts the distribution", [("0.0 if done else gamma ** len(rewards)", "gamma ** len(rewards)")]),
    ("3-rainbow-target", "action chosen by the minimum mean", [("np.argmax(np.asarray(p_online) @ z)", "np.argmin(np.asarray(p_online) @ z)")]),
    ("3-rainbow-target", "priority drops the exponent", [("return max(kl, 0.0) ** omega", "return max(kl, 0.0)")]),
    ("3-rainbow-target", "KL in the wrong direction", [("m[mask] * np.log(m[mask] / p_pred[mask])", "p_pred[mask] * np.log(p_pred[mask] / m[mask])")]),
    ("4-dueling-and-replay-weights", "advantage mean over atoms", [("adv.mean(axis=0, keepdims=True)", "adv.mean(axis=1, keepdims=True)")]),
    ("4-dueling-and-replay-weights", "advantage mean not subtracted", [(" - adv.mean(axis=0, keepdims=True)", "")]),
    ("4-dueling-and-replay-weights", "softmax over actions", [("logits.max(axis=1, keepdims=True)", "logits.max(axis=0, keepdims=True)"), ("e.sum(axis=1, keepdims=True)", "e.sum(axis=0, keepdims=True)")]),
    ("4-dueling-and-replay-weights", "softmax without max subtraction", [("np.exp(logits - logits.max(axis=1, keepdims=True))", "np.exp(logits)")]),
    ("4-dueling-and-replay-weights", "priorities not normalized", [("return p / p.sum()", "return p")]),
    ("4-dueling-and-replay-weights", "omega ignored", [("** omega", "** 1.0")]),
    ("4-dueling-and-replay-weights", "positive exponent on the weights", [("** (-beta)", "** beta")]),
    ("4-dueling-and-replay-weights", "weights not divided by the maximum", [("return w / w.max()", "return w")]),
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
