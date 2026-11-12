"""The reconstruction: learn a team size under different deadlines (ours)."""

import numpy as np

from swarm_lib import progress, finish_time, train, mean_n

print("Fixed deadline, 300 steps, group 16, lr 0.5, seed 0")
for tmax in [2, 5, 10, 20, 40]:
    best = max(range(1, 9), key=lambda n: progress(n, tmax))
    pr = train(lambda n, s, t=tmax: progress(n, t))
    print(f"Tmax={tmax:>3}  best n={best}  learned E[n]={mean_n(pr):.2f}  mode={int(pr.argmax()) + 1}")

print()
print("Variants")
pr = train(lambda n, s: 1.0)
print(f"no deadline, completion only   E[n]={mean_n(pr):.2f}  (uniform is 4.50)")
pr = train(lambda n, s: 1.0 - 0.5 * finish_time(n) / 100.0)
print(f"explicit latency penalty       E[n]={mean_n(pr):.2f}")
pr = train(lambda n, s: progress(n, 2 + 18 * s / 299))
print(f"deadline ramped 2 -> 20        E[n]={mean_n(pr):.2f}")
ramp = [round(mean_n(train(lambda n, s: progress(n, 2 + 18 * s / 299), seed=sd)), 2) for sd in range(6)]
print("ramp, seeds 0 to 5             E[n]=", ramp)
for n in range(1, 9):
    print(f"  finish time n={n}: {finish_time(n):.1f}")
