#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/deepseek-v2/scripts/check_mutants.py
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
    ("1-kv-cache-size", "MLA cache counts a factor of 2 (keys and values)", [("return (d_c + d_hR) * layers", "return 2 * (d_c + d_hR) * layers")]),
    ("1-kv-cache-size", "MLA cache forgets the decoupled RoPE key", [("return (d_c + d_hR) * layers", "return d_c * layers")]),
    ("1-kv-cache-size", "MLA cache forgets the layer count", [("return (d_c + d_hR) * layers", "return d_c + d_hR")]),
    ("1-kv-cache-size", "MHA cache forgets the factor of 2", [("return 2 * n_h * d_h * layers", "return n_h * d_h * layers")]),
    ("1-kv-cache-size", "equivalent groups divides by d_h, not 2 d_h", [("(d_c + d_hR) / (2 * d_h)", "(d_c + d_hR) / d_h")]),
    ("2-absorb-up-projection", "absorbed scale uses sqrt(d_c)", [("scores = q_abs @ C.T / np.sqrt(d_h)", "scores = q_abs @ C.T / np.sqrt(C.shape[-1])")]),
    ("2-absorb-up-projection", "absorbed skips W_UV (returns latent context)", [("out[:, i] = ctx @ WUV_i.T", "out[:, i] = ctx[:, :d_h]")]),
    ("2-absorb-up-projection", "absorbed uses W_UV for the query side", [("q_abs = Q[:, i] @ WUK_i", "q_abs = Q[:, i] @ WUV_i")]),
    ("2-absorb-up-projection", "absorbed has no causal mask", [("softmax(_causal(scores)) @ C", "softmax(scores) @ C")]),
    ("2-absorb-up-projection", "naive scale uses sqrt(d_c)", [("scores = Q[:, i] @ K[:, i].T / np.sqrt(d_h)", "scores = Q[:, i] @ K[:, i].T / np.sqrt(C.shape[-1])")]),
    ("2-absorb-up-projection", "naive reshapes K without grouping by head", [("K = (C @ WUK.T).reshape(T, n_h, d_h)", "K = (C @ WUK.T).reshape(n_h, T, d_h).transpose(1, 0, 2)")]),
    ("3-decoupled-rope", "RoPE also applied to the content key", [("content = np.einsum(\"tid,jid->itj\", qC, kC)", "content = np.einsum(\"tid,jid->itj\", qC, rope(kC, pos))")]),
    ("3-decoupled-rope", "scale uses sqrt(d_h) only", [("/ np.sqrt(d_h + d_hR)", "/ np.sqrt(d_h)")]),
    ("3-decoupled-rope", "RoPE on the query only", [("kR_rot = rope(kR, pos) ", "kR_rot = kR ")]),
    ("3-decoupled-rope", "rope pairs halves, not adjacent coordinates", [("a, b = x[..., 0::2], x[..., 1::2]", "a, b = x[..., : d // 2], x[..., d // 2 :]"), ("out[..., 0::2] = a * cos - b * sin", "out[..., : d // 2] = a * cos - b * sin"), ("out[..., 1::2] = a * sin + b * cos", "out[..., d // 2 :] = a * sin + b * cos")]),
    ("3-decoupled-rope", "rope sign flipped", [("out[..., 0::2] = a * cos - b * sin", "out[..., 0::2] = a * cos + b * sin"), ("out[..., 1::2] = a * sin + b * cos", "out[..., 1::2] = -a * sin + b * cos")]),
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
