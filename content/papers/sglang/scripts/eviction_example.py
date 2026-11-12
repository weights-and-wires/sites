"""LRU eviction of leaves with reference counts (Section 3).

Run from the paper folder: python3 scripts/eviction_example.py

The cache class is the solution to exercise 2. Tokens are integers and the
requests are this page's own toy example.
"""
import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lru_solution", HERE.parent / "exercises" / "2-lru-eviction" / "solution.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def show(c, label):
    rows = []

    def walk(node, depth):
        for first in sorted(node.children):
            ch = node.children[first]
            rows.append(f"{'  ' * depth}{list(ch.edge)}  t={ch.last_access} ref={ch.ref}")
            walk(ch, depth + 1)

    walk(c.root, 0)
    print(f"{label}: size {c.size()}")
    print("\n".join(rows))


c = mod.RadixCache(100)
X, Y, Z = (1, 2, 3, 4), (1, 2, 5, 6), (9, 9, 9)
for seq in (X, Y, Z):
    c.insert(seq)
show(c, "after inserting X, Y, Z")

c.lock(Y)
print("\nlock Y (a running request)")
freed = c.evict(3)
print(f"evict(3) freed {freed}")
show(c, "after eviction")

c.unlock(Y)
freed = c.evict(3)
print(f"\nunlock Y, evict(3) freed {freed}")
show(c, "after eviction")
