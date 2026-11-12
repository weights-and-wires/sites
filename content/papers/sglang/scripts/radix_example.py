"""A radix tree over tokens: insertion, node splitting, and prefix matching.

Run from the paper folder: python3 scripts/radix_example.py

Tokens are words. The requests are this page's own toy example, not the paper's
workload. The tree class is the solution to exercise 1.
"""
import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def load(name, folder):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "exercises" / folder / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tree_mod = load("radix_solution", "1-radix-tree")
tree = tree_mod.RadixTree()


def edges(t):
    out = []

    def walk(node, depth):
        for first in sorted(node.children):
            child = node.children[first]
            out.append("  " * depth + " ".join(child.edge))
            walk(child, depth + 1)

    walk(t.root, 0)
    return "\n".join(out)


SYSTEM = "You are helpful".split()
requests = [
    ("chat 1, turn 1", SYSTEM + "Hi there".split()),
    ("chat 2, turn 1", SYSTEM + "What is KV".split()),
    ("chat 1, turn 2", SYSTEM + "Hi there Bye now".split()),
    ("few-shot query", "Q1 A1 Q2".split()),
]

total_without = 0
for name, req in requests:
    new = tree.insert(req)
    total_without += len(req)
    print(f"insert {name}: {len(req)} tokens, {new} new, {tree.num_nodes()} nodes, {tree.num_tokens()} stored")
    print(edges(tree))

print()
print(f"tokens in the four requests: {total_without}; tokens stored in the tree: {tree.num_tokens()}")

print()
for q in ["You are helpful Hi there Bye", "You are helpful What", "You are", "Q1 A1 Q9", "Zebra"]:
    print(f"match {q!r}: {tree.match(q.split())} of {len(q.split())} tokens")

print()
t2 = tree_mod.RadixTree()
print("insert [a b c d]:", t2.insert(list("abcd")), "new,", t2.num_nodes(), "node")
print("insert [a b c e]:", t2.insert(list("abce")), "new,", t2.num_nodes(), "nodes")
print("insert [a b]    :", t2.insert(list("ab")), "new,", t2.num_nodes(), "nodes")
print(edges(t2))
