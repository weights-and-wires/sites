"""Arithmetic on the paper's tables.

Table 1 (100-game matches, AlphaZero's wins/draws/losses), Table S3 (training
statistics), and Table S4 (positions searched per second). The Elo gap uses the
paper's logistic model, p = 1 / (1 + exp(c_elo (e(b) - e(a)))) with
c_elo = 1/400, which is the standard 10^(d/400) form. A draw counts as half a
point. Elo gaps from short matches are rough, and the paper notes that draws
compress the chess scale (Figure 2, footnote 3).

Run: python3 scripts/paper_numbers.py
"""

import math

# (white-as-AlphaZero W, D, L) and (black-as-AlphaZero W, D, L), AlphaZero's view
TABLE1 = {
    "chess vs Stockfish": ((25, 25, 0), (3, 47, 0)),
    "shogi vs Elmo": ((43, 2, 5), (47, 0, 3)),
    "Go vs AlphaGo Zero 3-day": ((31, 0, 19), (29, 0, 21)),
}
# games, minibatches, batch size
TABLE_S3 = {"chess": (44e6, 700_000, 9), "shogi": (24e6, 700_000, 12), "Go": (21e6, 700_000, 34)}
BATCH = 4096


def elo_gap(score):
    return 400 * math.log10(score / (1 - score))


if __name__ == "__main__":
    for name, (a, b) in TABLE1.items():
        w, d, l = (a[i] + b[i] for i in range(3))
        score = (w + 0.5 * d) / (w + d + l)
        gap = f"{elo_gap(score):+.0f}" if 0 < score < 1 else "n/a"
        print(f"{name}: {w} W, {d} D, {l} L, score {score:.3f}, Elo gap about {gap}")

    print()
    for game, (games, steps, hours) in TABLE_S3.items():
        samples = steps * BATCH
        print(f"{game}: {steps * BATCH / 1e9:.2f} billion sampled positions, "
              f"{samples / games:.0f} per self-play game, {steps / (hours * 3600):.1f} steps/s")

    print()
    print(f"chess search speed: Stockfish {70_000_000 / 80_000:.0f}x more positions per second than AlphaZero")
    print(f"shogi search speed: Elmo {35_000_000 / 40_000:.0f}x more positions per second than AlphaZero")
    print(f"chess: surpassed Stockfish at 300k of 700k steps = {300 / 700:.0%} of the run")
    print(f"Go: 165k steps of 700k = {165 / 700:.0%} to pass AlphaGo Lee")
    # Input and policy sizes from Tables S1 and S2: N x N x (M T + L) with T = 8 history steps.
    print()
    print(f"Go input planes:    2 x 8 + 1  = {2 * 8 + 1}")
    print(f"chess input planes: 14 x 8 + 7 = {14 * 8 + 7}")
    print(f"shogi input planes: 45 x 8 + 2 = {45 * 8 + 2}")
    print(f"chess policy size:  8 x 8 x 73  = {8 * 8 * 73} moves")
    print(f"shogi policy size:  9 x 9 x 139 = {9 * 9 * 139} moves")
    print(f"Go policy size:     19 x 19 + 1 = {19 * 19 + 1} moves")
    # Elo to win probability, used by the self-check
    for d in (100, 200, 400):
        print(f"Elo gap {d}: expected score {1 / (1 + 10 ** (-d / 400)):.3f}")
