"""Block counts for parallel sampling, beam search, and a shared prefix.

Run from the paper folder: python3 scripts/sharing_example.py

Memory saving follows Section 6.3: blocks saved by sharing divided by the
blocks needed without sharing. The prompt, output, and beam sizes are this
page's own, not the paper's workload.
"""
import math

B = 16


def blocks(n_tokens):
    return math.ceil(n_tokens / B)


print("Parallel sampling: prompt P tokens, T output tokens per sample, n samples, B =", B)
for P, T, n in [(100, 50, 4), (1000, 50, 4), (1000, 500, 4)]:
    no_share = n * blocks(P + T)
    full_prompt_blocks = P // B                 # blocks that hold only prompt tokens
    shared = full_prompt_blocks + n * (blocks(P + T) - full_prompt_blocks)
    print(f"  P={P:>4} T={T:>3} n={n}: without sharing {no_share:>4}, with sharing {shared:>4}, saving {1 - shared / no_share:.1%}")

print("\nA 1000-token prompt whose length is not a multiple of B: the last prompt block is partial")
P, n = 1000, 4
print(f"  1000 = {P // B} full blocks + {P % B} tokens; the partial block is shared until the first write, then copied once per extra sample")
print(f"  copies at the first output token: {n - 1} (the last sample reuses the original)")

print("\nShared prefix (Figure 16 caption: prefixes of 80 and 341 tokens), B =", B)
for prefix in (80, 341):
    full = prefix // B
    partial = prefix % B
    tail = f"{partial} tokens fill a partial block marked copy-on-write" if partial else "no partial block"
    print(f"  prefix {prefix}: {full} full blocks shared; {tail}")
    for users in (1, 10, 100):
        saved = users * full - full if users > 0 else 0
        print(f"    {users:>3} requests: {saved} blocks saved of {users * (full + (partial > 0))} prefix blocks")

print("\nBeam search, k = 4, Figure 9: shared blocks after the step")
# From the text of Section 4.4: after the step, all candidates share blocks 0, 1, 3;
# candidates 0 and 1 share block 6; candidates 2 and 3 share block 7.
tables = {0: [0, 1, 3, 6, 9], 1: [0, 1, 3, 6, 10], 2: [0, 1, 3, 7, 11], 3: [0, 1, 3, 7, 12]}
logical = sum(len(t) for t in tables.values())
physical = len({b for t in tables.values() for b in t})
print(f"  logical blocks {logical}, physical blocks {physical}, saving {1 - physical / logical:.1%}")
print("  (the last blocks 9 to 12 are the new ones from the step; the assignment of 9 to 12 to candidates is this page's reading of the figure)")
