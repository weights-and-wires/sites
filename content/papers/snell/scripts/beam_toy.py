"""PRM beam search on a tiny step tree (Section 5.2).

The paper's beam search keeps N beams and a beam width M. Each round it keeps
the top N/M prefixes by PRM score, then samples M next steps from each.
This script traces N=4, M=2 on a two-step tree with hand-written PRM scores.

Run: python3 scripts/beam_toy.py
"""

N, M, STEPS = 4, 2, 2

# PRM score of every prefix. Prefixes are tuples of step indices.
SCORE = {
    (0,): 0.30, (1,): 0.80, (2,): 0.50, (3,): 0.90,
    (1, 0): 0.40, (1, 1): 0.95,
    (3, 0): 0.60, (3, 1): 0.70,
}


def expand(prefix, m):
    return [prefix + (i,) for i in range(m)]


def beam_search(n, m, steps):
    beams = expand((), n)                                  # N first steps
    for round_no in range(1, steps + 1):
        ranked = sorted(beams, key=lambda p: -SCORE[p])    # stable: ties keep order
        print(f"round {round_no}: scores " + ", ".join(f"{p}={SCORE[p]:.2f}" for p in beams))
        if round_no == steps:
            return ranked
        kept = ranked[: n // m]
        print(f"         keep top N/M = {n // m}: {kept}")
        beams = [child for p in kept for child in expand(p, m)]
        print(f"         expand each with M = {m}: {len(beams)} candidates")


final = beam_search(N, M, STEPS)
print(f"best final prefix: {final[0]} with score {SCORE[final[0]]:.2f}")
print(f"final candidates, best first: {final}")
