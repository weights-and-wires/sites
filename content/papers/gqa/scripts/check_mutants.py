#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/gqa/scripts/check_mutants.py
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
    ("1-kv-cache-bytes", "forgets the factor 2 for keys and values", [("return 2 * n_layers * n_kv_heads", "return n_layers * n_kv_heads")]),
    ("1-kv-cache-bytes", "ignores the number of layers", [("return 2 * n_layers * n_kv_heads * d_head * bytes_per_value", "return 2 * n_kv_heads * d_head * bytes_per_value")]),
    ("1-kv-cache-bytes", "ignores the default precision (4 bytes)", [("bytes_per_value=2):\n    \"\"\"Bytes added", "bytes_per_value=4):\n    \"\"\"Bytes added")]),
    ("1-kv-cache-bytes", "does not subtract the weights", [("(gpu_bytes - weight_bytes) // per_sequence", "gpu_bytes // per_sequence")]),
    ("1-kv-cache-bytes", "rounds to nearest instead of down", [("(gpu_bytes - weight_bytes) // per_sequence", "round((gpu_bytes - weight_bytes) / per_sequence)")]),
    ("2-grouped-query-attention", "interleaved grouping (tile instead of repeat)", [("K = np.repeat(split(X @ Wk, G), r, axis=0)", "K = np.tile(split(X @ Wk, G), (r, 1, 1))"), ("V = np.repeat(split(X @ Wv, G), r, axis=0)", "V = np.tile(split(X @ Wv, G), (r, 1, 1))")]),
    ("2-grouped-query-attention", "values use the interleaved grouping, keys use the right one", [("V = np.repeat(split(X @ Wv, G), r, axis=0)", "V = np.tile(split(X @ Wv, G), (r, 1, 1))")]),
    ("2-grouped-query-attention", "mask not passed to attention", [("scaled_dot_product_attention(Q, K, V, mask)\n    concat", "scaled_dot_product_attention(Q, K, V)\n    concat")]),
    ("2-grouped-query-attention", "forget W^O", [("return concat @ Wo", "return concat")]),
    ("2-grouped-query-attention", "reshape to (heads, n, d_k) without transpose", [("M.reshape(n, heads, d_k).transpose(1, 0, 2)", "M.reshape(heads, n, d_k)")]),
    ("2-grouped-query-attention", "keys built from the value projection", [("split(X @ Wk, G)", "split(X @ Wv, G)")]),
    ("3-mean-pool-checkpoint", "sum instead of mean", [(".mean(axis=2)", ".sum(axis=2)")]),
    ("3-mean-pool-checkpoint", "first head of each group instead of the mean", [(".mean(axis=2)", "[:, :, 0, :]")]),
    ("3-mean-pool-checkpoint", "interleaved groups", [("W.reshape(d_model, G, H // G, d_k).mean(axis=2)", "W.reshape(d_model, H // G, G, d_k).mean(axis=1)")]),
    ("3-mean-pool-checkpoint", "average over all heads for every group", [("W.reshape(d_model, G, H // G, d_k).mean(axis=2).reshape(d_model, G * d_k)", "np.tile(W.reshape(d_model, H, d_k).mean(axis=1), (1, G))")]),
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
