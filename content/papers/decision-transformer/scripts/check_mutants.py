#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/decision-transformer/scripts/check_mutants.py
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
# Known gap: exercise 5 can't catch a softmax over the query axis, because the
# model still learns modular addition with it. The NumPy exercises cover that bug.
MUTANTS = [
    ("1-returns-to-go", "forward cumulative sum (reward so far)", [("np.cumsum(r[..., ::-1], axis=-1)[..., ::-1]", "np.cumsum(r, axis=-1)")]),
    ("1-returns-to-go", "exclusive sum (current reward left out)", [("return np.cumsum(r[..., ::-1], axis=-1)[..., ::-1]", "g = np.cumsum(r[..., ::-1], axis=-1)[..., ::-1]\n    return g - r")]),
    ("1-returns-to-go", "reverses the batch axis, not time", [("np.cumsum(r[..., ::-1], axis=-1)[..., ::-1]", "np.cumsum(r[::-1], axis=-1)[::-1]")]),
    ("1-returns-to-go", "episode total repeated at every step", [("return np.cumsum(r[..., ::-1], axis=-1)[..., ::-1]", "return np.broadcast_to(r.sum(axis=-1, keepdims=True), r.shape).copy()")]),
    ("1-returns-to-go", "integer input keeps an integer dtype", [("np.asarray(rewards, dtype=float)", "np.asarray(rewards)")]),
    ("2-token-sequence", "state placed before the return", [("np.stack([R_emb + t_emb, s_emb + t_emb, a_emb + t_emb], axis=1)", "np.stack([s_emb + t_emb, R_emb + t_emb, a_emb + t_emb], axis=1)")]),
    ("2-token-sequence", "action placed before the state (leaks a_t)", [("np.stack([R_emb + t_emb, s_emb + t_emb, a_emb + t_emb], axis=1)", "np.stack([R_emb + t_emb, a_emb + t_emb, s_emb + t_emb], axis=1)")]),
    ("2-token-sequence", "modalities concatenated in blocks, not interleaved", [("stacked = np.stack([R_emb + t_emb, s_emb + t_emb, a_emb + t_emb], axis=1)\n    return stacked.reshape(3 * K, d)", "return np.concatenate([R_emb + t_emb, s_emb + t_emb, a_emb + t_emb], axis=0)")]),
    ("2-token-sequence", "timestep embedding only on the state token", [("np.stack([R_emb + t_emb, s_emb + t_emb, a_emb + t_emb], axis=1)", "np.stack([R_emb, s_emb + t_emb, a_emb], axis=1)")]),
    ("2-token-sequence", "state indices point at the return tokens", [("np.arange(K) * 3 + 1", "np.arange(K) * 3")]),
    ("3-rollout-loop", "target is never decremented", [("R.append(R[-1] - r)", "R.append(R[-1])")]),
    ("3-rollout-loop", "reward added instead of subtracted", [("R.append(R[-1] - r)", "R.append(R[-1] + r)")]),
    ("3-rollout-loop", "decrement restarts from the original target", [("R.append(R[-1] - r)", "R.append(float(target_return) - r)")]),
    ("3-rollout-loop", "context is not truncated to K", [("n = min(K, len(R))", "n = len(R)")]),
    ("3-rollout-loop", "timesteps renumbered inside the window", [("t_ctx = np.array(t[-n:])", "t_ctx = np.arange(1, n + 1)")]),
    ("3-rollout-loop", "a[-(n - 1):] slice returns every action when n is 1", [("a_ctx = np.array(a[len(a) - (n - 1):])", "a_ctx = np.array(a[-(n - 1):])")]),
    ("3-rollout-loop", "action context as long as the state context", [("a_ctx = np.array(a[len(a) - (n - 1):])", "a_ctx = np.array((a + [0])[len(a) + 1 - n:])")]),
    ("4-conditioned-policy", "policy ignores the return (behavior cloning)", [("c = table[s, int(np.clip(R, R_MIN, 0)) - R_MIN]", "c = table[s].sum(axis=0)")]),
    ("4-conditioned-policy", "return index off by one when counting", [("g_idx = np.clip(np.asarray(G), R_MIN, 0).astype(int) - R_MIN", "g_idx = np.clip(np.asarray(G), R_MIN, 0).astype(int) - R_MIN - 1")]),
    ("4-conditioned-policy", "no clipping of low returns (negative index wraps)", [("np.clip(np.asarray(G), R_MIN, 0).astype(int)", "np.asarray(G).astype(int)")]),
    ("4-conditioned-policy", "fancy-index += counts repeated triples once", [("np.add.at(table, (np.asarray(S), g_idx, np.asarray(A)), 1)", "table[np.asarray(S), g_idx, np.asarray(A)] += 1")]),
    ("4-conditioned-policy", "no fallback for unseen (s, R)", [("    if total == 0:\n        return np.array([0.5, 0.5])\n", "")]),
    ("4-conditioned-policy", "normalizes by the table total, not the cell total", [("return c / total", "return c / table.sum()")]),
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
