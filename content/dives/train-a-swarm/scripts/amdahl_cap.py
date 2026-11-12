"""Worked example C: Amdahl's law plus a setup tax, then a wall-clock cap."""

from swarm_lib import speed, progress

print("speed(n) with p = 0.9:")
for n in [1, 4, 8]:
    print(f"  n={n}: {speed(n):.2f}")
print()
print("progress by the cap (setup 0.6 per extra agent):")
for tmax in [5, 20]:
    row = {n: round(progress(n, tmax), 1) for n in [1, 4, 8]}
    print(f"  Tmax={tmax:>2}: {row}")
print()
print("crossover time, when n agents first beat 1 agent:")
for n in [2, 4, 8]:
    s = speed(n)
    t = s * 0.6 * (n - 1) / (s - 1)
    print(f"  n={n}: t* = {t:.2f}")
