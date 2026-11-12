#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/sglang/scripts/check_mutants.py
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
    ("1-radix-tree", "match counts only whole edges", [("            i += k\n            if k < len(child.edge):\n                break\n            node = child\n        return i", "            if k < len(child.edge):\n                break\n            i += k\n            node = child\n        return i")]),
    ("1-radix-tree", "match ignores the end of the query", [("and i + k < len(tokens) and child.edge[k] == tokens[i + k]:\n                k += 1\n            i += k\n            if k < len(child.edge):\n                break", "and child.edge[k] == tokens[min(i + k, len(tokens) - 1)]:\n                k += 1\n            i += k\n            if k < len(child.edge):\n                break")]),
    ("1-radix-tree", "insert returns the length instead of the new tokens", [("return len(tokens) - i", "return len(tokens)")]),
    ("1-radix-tree", "insert never splits (new leaf under the old edge key)", [("            if k < len(child.edge):\n                mid = Node(child.edge[:k], node)", "            if False:\n                mid = Node(child.edge[:k], node)")]),
    ("1-radix-tree", "split forgets the parent link", [("                child.parent = mid\n", "")]),
    ("1-radix-tree", "split keeps the old first-token key for the child", [("mid.children[child.edge[0]] = child", "mid.children[tokens[i]] = child")]),
    ("1-radix-tree", "an ending-inside-an-edge sequence is not split", [("            if k < len(child.edge):\n                mid = Node(child.edge[:k], node)", "            if k < len(child.edge) and i + k < len(tokens):\n                mid = Node(child.edge[:k], node)")]),
    ("2-lru-eviction", "evicts the most recently used leaf", [("min(leaves, key=lambda n: n.last_access)", "max(leaves, key=lambda n: n.last_access)")]),
    ("2-lru-eviction", "ignores reference counts", [("and not n.children and n.ref == 0]", "and not n.children]")]),
    ("2-lru-eviction", "evicts interior nodes too", [("and not n.children and n.ref == 0]", "and n.ref == 0]")]),
    ("2-lru-eviction", "does not cascade to parents", [("leaves = [n for n in self._nodes() if", "leaves = [n for n in self._leaves0 if") , ("        freed = 0\n        while freed < num_tokens:", "        if not hasattr(self, '_leaves0'):\n            self._leaves0 = [n for n in self._nodes() if n is not self.root and not n.children]\n        self._leaves0 = [n for n in self._leaves0 if n.parent is not None and n.edge[0] in n.parent.children]\n        freed = 0\n        while freed < num_tokens:")]),
    ("2-lru-eviction", "counts nodes instead of tokens", [("freed += len(victim.edge)", "freed += 1")]),
    ("2-lru-eviction", "stops after one leaf", [("            freed += len(victim.edge)", "            freed += len(victim.edge)\n            break")]),
    ("2-lru-eviction", "loops over leaves in tree order, not by recency", [("victim = min(leaves, key=lambda n: n.last_access)", "victim = leaves[0]")]),
    ("3-cache-aware-scheduling", "lpm picks the shortest match", [("if m > best_len:", "if best_len < 0 or m < best_len:")]),
    ("3-cache-aware-scheduling", "lpm breaks ties by the latest arrival", [("if m > best_len:", "if m >= best_len:")]),
    ("3-cache-aware-scheduling", "fcfs picks the newest", [('    if policy == "fcfs":\n        return 0', '    if policy == "fcfs":\n        return len(waiting) - 1')]),
    ("3-cache-aware-scheduling", "serve picks once and keeps the order", [("        pos = pick_next([r for _, r in waiting], cache, policy)", "        pos = 0 if order else pick_next([r for _, r in waiting], cache, policy)")]),
    ("3-cache-aware-scheduling", "serve undercounts the total", [("total += len(req)", "total += len(req) - 1")]),
    ("3-cache-aware-scheduling", "serve drops the hits", [("hits += cache.admit(req)", "cache.admit(req)")]),
    ("3-cache-aware-scheduling", "serve reports positions, not original indices", [("order.append(idx)", "order.append(pos)")]),
    ("4-jump-forward", "jump ignores accepting states", [(" and state not in fsm.accepting:", ":")]),
    ("4-jump-forward", "jump goes through branches", [("while len(fsm.edges[state]) == 1", "while len(fsm.edges[state]) >= 1")]),
    ("4-jump-forward", "jump drops the collected characters", [("        text.append(ch)", "        text.append(\"\")")]),
    ("2-lru-eviction", "split gives the new parent no reference count", [("                mid.ref = child.ref                  # every user of the child also uses mid\n", "")]),
    ("4-jump-forward", "decode counts forced characters as steps", [("        text, state = jump_forward(fsm, state)\n        out.append(text)", "        text, state = jump_forward(fsm, state)\n        steps += len(text)\n        out.append(text)")]),
    ("4-jump-forward", "decode never offers a stop", [("+ ([None] if state in fsm.accepting else [])\n        ch = choose(options)\n        steps += 1\n        if ch is None:\n            break\n        out.append(ch)\n        state = edges[ch]\n    return \"\".join(out), steps\n\n\ndef decode_naive", "\n        ch = choose(options)\n        steps += 1\n        if ch is None:\n            break\n        out.append(ch)\n        state = edges[ch]\n    return \"\".join(out), steps\n\n\ndef decode_naive")]),
    ("4-jump-forward", "decode forgets the step that chose to stop", [("        steps += 1\n        if ch is None:\n            break\n        out.append(ch)\n        state = edges[ch]\n    return \"\".join(out), steps\n\n\ndef decode_naive", "        if ch is None:\n            break\n        steps += 1\n        out.append(ch)\n        state = edges[ch]\n    return \"\".join(out), steps\n\n\ndef decode_naive")]),
    ("4-jump-forward", "decode loses the forced text", [("        text, state = jump_forward(fsm, state)\n        out.append(text)", "        text, state = jump_forward(fsm, state)")]),
    ("4-jump-forward", "decode passes unsorted options", [("options = sorted(edges) +", "options = list(reversed(sorted(edges))) +")]),
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
    skip_torch = False
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
