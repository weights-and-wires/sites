"""Compare request-level and iteration-level scheduling.

Run from the paper folder: python3 scripts/static_vs_continuous.py
"""

import numpy as np
from simulate import continuous_batching, make_workload, static_batching, utilization

print("Tiny example: arrivals [0,0,1,2], lengths [2,6,2,2], max_bs 2")
arr, lens = [0, 0, 1, 2], [2, 6, 2, 2]
for name, fn in [("request-level  ", static_batching), ("iteration-level", continuous_batching)]:
    fin, use = fn(arr, lens, 2)
    lat = [f - a for f, a in zip(fin, arr)]
    print(f"{name} finish={fin} latency={lat} mean_latency={np.mean(lat):.2f} "
          f"iterations={len(use)} useful_per_iter={use} utilization={utilization(use, 2):.3f}")

for label, rate in [("all 200 requests queued at t=0", None), ("Poisson arrivals, 0.10 per iteration", 0.10)]:
    arr, lens = make_workload(200, rate, seed=0)
    print()
    print(f"Random workload, lengths U(1,128), max_bs 8, {label}")
    print(f"mean length {np.mean(lens):.2f}, total tokens {sum(lens)}")
    for name, fn in [("request-level  ", static_batching), ("iteration-level", continuous_batching)]:
        fin, use = fn(arr, lens, 8)
        lat = np.array(fin) - np.array(arr)
        print(f"{name} makespan={len(use)} tokens/iter={sum(lens) / len(use):.3f} "
              f"utilization={utilization(use, 8):.3f} mean_latency={lat.mean():.1f} "
              f"median_latency={np.median(lat):.1f}")

print()
print("Utilization over the first 20 iterations of the queued workload (useful slots of 8)")
arr, lens = make_workload(200, None, seed=0)
for name, fn in [("request-level  ", static_batching), ("iteration-level", continuous_batching)]:
    print(name, fn(arr, lens, 8)[1][:20])
