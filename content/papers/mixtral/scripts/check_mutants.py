#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/mixtral/scripts/check_mutants.py
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
    ("1-param-counts", "K and V projections use n_heads, not n_kv_heads", [("2 * d * cfg[\"n_kv_heads\"] * cfg[\"head_dim\"]", "2 * d * cfg[\"n_heads\"] * cfg[\"head_dim\"]")]),
    ("1-param-counts", "tied embeddings (counted once)", [("2 * cfg[\"vocab_size\"] * d + d", "cfg[\"vocab_size\"] * d + d")]),
    ("1-param-counts", "active counts one expert instead of top_k", [("cfg[\"top_k_experts\"] * expert", "expert")]),
    ("1-param-counts", "active counts every expert", [("cfg[\"top_k_experts\"] * expert", "cfg[\"num_experts\"] * expert")]),
    ("1-param-counts", "expert has two matrices, not three", [("expert = 3 * d", "expert = 2 * d")]),
    ("1-param-counts", "router forgotten", [("router = d * cfg[\"num_experts\"] if cfg[\"num_experts\"] > 1 else 0", "router = 0")]),
    ("1-param-counts", "dense model still gets a router", [("router = d * cfg[\"num_experts\"] if cfg[\"num_experts\"] > 1 else 0", "router = d * cfg[\"num_experts\"]")]),
    ("2-top2-gate", "softmax over all experts, then keep the top 2 (no renormalization)", [(
        "    top = np.take_along_axis(logits, idx, axis=-1)\n    e = np.exp(top - top.max(axis=-1, keepdims=True))\n    return idx, e / e.sum(axis=-1, keepdims=True)",
        "    e = np.exp(logits - logits.max(axis=-1, keepdims=True))\n    p = e / e.sum(axis=-1, keepdims=True)\n    return idx, np.take_along_axis(p, idx, axis=-1)")]),
    ("2-top2-gate", "picks the two smallest logits", [("np.argsort(-logits", "np.argsort(logits")]),
    ("2-top2-gate", "softmax over the token axis", [("top.max(axis=-1, keepdims=True)", "top.max(axis=0, keepdims=True)"), ("e / e.sum(axis=-1, keepdims=True)", "e / e.sum(axis=0, keepdims=True)")]),
    ("2-top2-gate", "no max subtraction (overflow)", [("np.exp(top - top.max(axis=-1, keepdims=True))", "np.exp(top)")]),
    ("2-top2-gate", "indices left unsorted by logit", [("idx = np.argsort(-logits, axis=-1, kind=\"stable\")[:, :k]", "idx = np.sort(np.argsort(-logits, axis=-1, kind=\"stable\")[:, :k], axis=-1)")]),
    ("3-moe-layer", "weights not renormalized (full softmax mass)", [(
        "    idx, weights = top_k_gate(x @ Wg, k)\n",
        "    idx, _ = top_k_gate(x @ Wg, k)\n    z = x @ Wg\n    p = np.exp(z - z.max(-1, keepdims=True))\n    p /= p.sum(-1, keepdims=True)\n    weights = np.take_along_axis(p, idx, axis=-1)\n")]),
    ("3-moe-layer", "every expert runs on every token, masked by weight", [(
        "        rows, slot = np.where(idx == e)\n        if rows.size == 0:\n            continue\n        out[rows] += weights[rows, slot][:, None] * expert(x[rows])",
        "        rows = np.arange(x.shape[0])\n        w = np.where(idx == e, weights, 0.0).sum(axis=1)\n        out += w[:, None] * expert(x)")]),
    ("3-moe-layer", "uses the weight of the other slot", [("weights[rows, slot][:, None]", "weights[rows, 1 - slot][:, None]")]),
    ("3-moe-layer", "overwrites instead of accumulating", [("out[rows] += ", "out[rows] = ")]),
    ("3-moe-layer", "ignores the router and always picks the first k experts", [("idx, weights = top_k_gate(x @ Wg, k)", "idx, weights = top_k_gate(x @ Wg, k)\n    idx = np.zeros_like(idx) + np.arange(k)")]),
    ("4-routing-locality", "compares the same slot only", [("overlap = (idx[1:, :, None] == idx[:-1, None, :]).any(axis=(1, 2))", "overlap = (idx[1:] == idx[:-1]).any(axis=1)")]),
    ("4-routing-locality", "counts matches instead of testing for any", [("overlap = (idx[1:, :, None] == idx[:-1, None, :]).any(axis=(1, 2))", "overlap = (idx[1:, :, None] == idx[:-1, None, :]).sum(axis=(1, 2))")]),
    ("4-routing-locality", "compares tokens two apart", [("idx[1:, 0] == idx[:-1, 0]", "idx[2:, 0] == idx[:-2, 0]"), ("(idx[1:, :, None] == idx[:-1, None, :])", "(idx[2:, :, None] == idx[:-2, None, :])")]),
    ("4-routing-locality", "either = second choice only", [("overlap = (idx[1:, :, None] == idx[:-1, None, :]).any(axis=(1, 2))", "overlap = idx[1:, 1] == idx[:-1, 1]")]),
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
