# Progress by the deadline for a team of n agents (our toy model, not the post's).
# Edit p (parallel fraction) or setup (minutes of coordination per extra agent).
p = 0.9
setup = 0.6


def speed(n):
    return 1.0 / ((1 - p) + p / n)  # Amdahl's law


def progress(n, tmax):
    return speed(n) * max(0.0, tmax - setup * (n - 1))


print("tmax   " + "".join(f"n={n:<6}" for n in range(1, 9)) + "best")
for tmax in [2, 5, 10, 20, 40]:
    row = [progress(n, tmax) for n in range(1, 9)]
    best = row.index(max(row)) + 1
    print(f"{tmax:<7}" + "".join(f"{v:<8.1f}" for v in row) + str(best))
