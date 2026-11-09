#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/switch/scripts/check_mutants.py
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
    ("1-top1-router", "softmax over the token axis", [("axis=-1, keepdims=True)\n    e = np.exp(shifted)\n    p = e / np.sum(e, axis=-1, keepdims=True)", "axis=0, keepdims=True)\n    e = np.exp(shifted)\n    p = e / np.sum(e, axis=0, keepdims=True)")]),
    ("1-top1-router", "softmax without the max subtraction", [("shifted = h - np.max(h, axis=-1, keepdims=True)", "shifted = h")]),
    ("1-top1-router", "gate is the largest logit, not the probability", [("gate = p[np.arange(len(p)), idx]", "gate = h[np.arange(len(h)), idx]")]),
    ("1-top1-router", "gate fixed at 1 (router never scales the output)", [("gate = p[np.arange(len(p)), idx]", "gate = np.ones(len(p))")]),
    ("1-top1-router", "choose the smallest probability", [("idx = np.argmax(p, axis=-1)", "idx = np.argmin(p, axis=-1)")]),
    ("2-load-balancing-loss", "f computed from probabilities instead of argmax", [("f = np.bincount(idx, minlength=N) / T", "f = p.mean(axis=0)")]),
    ("2-load-balancing-loss", "P computed from hard assignments", [("P = p.mean(axis=0)", "P = np.bincount(idx, minlength=N) / T")]),
    ("2-load-balancing-loss", "forget the factor N", [("return alpha * N * float(np.sum(f * P))", "return alpha * float(np.sum(f * P))")]),
    ("2-load-balancing-loss", "sum over tokens instead of mean for P", [("P = p.mean(axis=0)", "P = p.sum(axis=0)")]),
    ("2-load-balancing-loss", "alpha ignored", [("return alpha * N *", "return 1e-2 * N *")]),
    ("3-capacity-dispatch", "capacity rounded down", [("int(math.ceil(T / N * capacity_factor))", "int(T / N * capacity_factor)")]),
    ("3-capacity-dispatch", "capacity ignores the capacity factor", [("T / N * capacity_factor", "T / N")]),
    ("3-capacity-dispatch", "position counted over all tokens, not per expert", [("keep[t] = seen[e] < capacity\n        seen[e] += 1", "keep[t] = t < capacity")]),
    ("3-capacity-dispatch", "off by one: capacity + 1 slots", [("seen[e] < capacity", "seen[e] <= capacity")]),
    ("3-capacity-dispatch", "later tokens win the slots", [("for t, e in enumerate(idx):", "for t, e in reversed(list(enumerate(idx))):")]),
    ("3-capacity-dispatch", "dropped tokens are zeroed instead of passed through", [("return x + update", "return (x + update) * keep[:, None]")]),
    ("3-capacity-dispatch", "gate not applied to the expert output", [("update = gate[:, None] * expert_out * keep[:, None]", "update = expert_out * keep[:, None]")]),
    ("4-router-gradient", "no gradient through the gate (router never learns)", [("dgate = np.sum(dy * out, axis=1)", "dgate = np.zeros(len(x))")]),
    ("4-router-gradient", "softmax backward skipped", [("dh = p * (dp - np.sum(dp * p, axis=1, keepdims=True))", "dh = dp")]),
    ("4-router-gradient", "softmax backward drops the normalization term", [("dh = p * (dp - np.sum(dp * p, axis=1, keepdims=True))", "dh = p * dp")]),
    ("4-router-gradient", "expert gradient ignores the gate value", [("d_out_act = dy * gate[:, None]", "d_out_act = dy")]),
    ("4-router-gradient", "forward ignores the gate value", [("return gate[:, None] * out,", "return out,")]),
    ("4-router-gradient", "every expert receives every token's gradient", [("m = idx == i", "m = idx >= 0")]),
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
