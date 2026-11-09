"""Replays Figure 6 (block table) and Figure 8 (copy-on-write) of the paper.

Run from the paper folder: python3 scripts/block_table_example.py

The free list is ordered 7, 1, 3, 0, 2, 4, 5, 6, 8 so that the allocator hands
out the same physical block IDs as the figures.
"""


class Allocator:
    def __init__(self, order):
        self.free = list(order)
        self.ref = {}

    def allocate(self):
        b = self.free.pop(0)
        self.ref[b] = 1
        return b

    def release(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            del self.ref[b]
            self.free.append(b)


B = 4
alloc = Allocator([7, 1, 3, 0, 2, 4, 5, 6, 8])


class Seq:
    def __init__(self):
        self.table = []      # list of [physical block, filled]

    def append(self):
        if not self.table or self.table[-1][1] == B:
            self.table.append([alloc.allocate(), 0])
        self.table[-1][1] += 1


def show(label, seq):
    print(f"{label}: " + ", ".join(f"logical {i} -> physical {p} (filled {f})" for i, (p, f) in enumerate(seq.table)))


print("Figure 6: a 7-token prompt, B = 4")
a = Seq()
for _ in range(7):
    a.append()
show("after prefill        ", a)
a.append()
show("1st decoding step    ", a)
a.append()
show("2nd decoding step    ", a)

print("\nFigure 8: two samples share a 7-token prompt")
alloc = Allocator([7, 1, 3, 0, 2, 4, 5, 6, 8])
a1 = Seq()
for _ in range(7):
    a1.append()
a2 = Seq()
a2.table = [list(e) for e in a1.table]
for p, _ in a2.table:
    alloc.ref[p] += 1
print("reference counts after fork:", dict(sorted(alloc.ref.items())))


def append_cow(seq):
    if seq.table and seq.table[-1][1] < B and alloc.ref[seq.table[-1][0]] > 1:
        old = seq.table[-1][0]
        new = alloc.allocate()
        alloc.release(old)
        print(f"  copy-on-write: block {old} -> block {new}, ref[{old}] now {alloc.ref[old]}")
        seq.table[-1][0] = new
    seq.append()


print("A1 writes its first output token")
append_cow(a1)
print("A2 writes its first output token")
append_cow(a2)
show("A1", a1)
show("A2", a2)
print("reference counts:", dict(sorted(alloc.ref.items())))
print("physical blocks in use:", len(alloc.ref), "; without sharing:", len(a1.table) + len(a2.table))
