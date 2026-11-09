#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/pagedattention/scripts/check_mutants.py
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

EAGER = (
    "        if self.num_tokens % self.block_size == 0:\n"
    "            self.blocks.append(self.allocator.allocate())\n"
    "        offset = self.num_tokens % self.block_size\n"
    "        self.num_tokens += 1\n"
    "        return self.blocks[-1], offset\n"
)
EAGER_BAD = (
    "        if not self.blocks:\n"
    "            self.blocks.append(self.allocator.allocate())\n"
    "        offset = self.num_tokens % self.block_size\n"
    "        self.num_tokens += 1\n"
    "        placed = self.blocks[-1]\n"
    "        if self.num_tokens % self.block_size == 0:\n"
    "            self.blocks.append(self.allocator.allocate())\n"
    "        return placed, offset\n"
)
PER_BLOCK_SOFTMAX = (
    "    s = np.concatenate(scores)\n"
    "    e = np.exp(s - s.max())\n"
    "    a = e / e.sum()\n"
    "    return a @ np.concatenate(values)"
)
PER_BLOCK_SOFTMAX_BAD = (
    "    out = 0\n"
    "    for s, v in zip(scores, values):\n"
    "        e = np.exp(s - s.max())\n"
    "        out = out + (e / e.sum()) @ v\n"
    "    return out / len(scores)"
)

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-block-table", "allocates the next block as soon as one fills (eager)", [(EAGER, EAGER_BAD)]),
    ("1-block-table", "free ignores the reference count", [("if self._ref[block] == 0:", "if True:")]),
    ("1-block-table", "freed block goes to the front of the free list", [("self._free.append(block)", "self._free.insert(0, block)")]),
    ("1-block-table", "translate has no range check", [("if not 0 <= pos < self.num_tokens:", "if pos < 0:")]),
    ("1-block-table", "release forgets to free the blocks", [("            self.allocator.free(block)\n        self.blocks = []", "            pass\n        self.blocks = []")]),
    ("1-block-table", "allocate never raises MemoryError", [("if not self._free:", "if False:")]),
    ("2-fragmentation", "block count is floor + 1", [("math.ceil(n_tokens / block_size)", "n_tokens // block_size + 1")]),
    ("2-fragmentation", "admission skips a misfit instead of stopping", [("            break\n        used += c", "            continue\n        used += c")]),
    ("2-fragmentation", "exact fit rejected (>= instead of >)", [("if used + c > capacity:", "if used + c >= capacity:")]),
    ("2-fragmentation", "paged cost counts blocks, not slots", [("return blocks_needed(n_tokens, block_size) * block_size", "return blocks_needed(n_tokens, block_size)")]),
    ("2-fragmentation", "contiguous waste ignores the used slots", [("max_len - n for n in lengths", "max_len for n in lengths")]),
    ("3-copy-on-write", "writes into a shared block without copying", [("elif self.allocator.ref_count(self.blocks[-1]) > 1:", "elif False:")]),
    ("3-copy-on-write", "copies even when the block has one owner", [("elif self.allocator.ref_count(self.blocks[-1]) > 1:", "elif self.allocator.ref_count(self.blocks[-1]) >= 1:")]),
    ("3-copy-on-write", "forgets to drop the reference to the old block", [("            self.allocator.free(old)\n", "")]),
    ("3-copy-on-write", "fork does not take references", [("            self.allocator.incref(block)\n        return child", "            pass\n        return child")]),
    ("4-paged-attention", "softmax per block, not across blocks", [(PER_BLOCK_SOFTMAX, PER_BLOCK_SOFTMAX_BAD)]),
    ("4-paged-attention", "reads unfilled slots of the last block", [("filled = min(B, n_tokens - j * B)", "filled = B")]),
    ("4-paged-attention", "uses the logical index as the physical block", [("pool_K[phys, :filled]", "pool_K[j, :filled]"), ("pool_V[phys, :filled]", "pool_V[j, :filled]")]),
    ("4-paged-attention", "no max subtraction before exp", [("np.exp(s - s.max())", "np.exp(s)")]),
    ("4-paged-attention", "no 1/sqrt(d) scaling", [(" @ q / np.sqrt(d)", " @ q")]),
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
