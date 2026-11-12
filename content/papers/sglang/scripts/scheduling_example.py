"""Cache-aware scheduling: longest-prefix-first against first-come-first-served.

Run from the paper folder: python3 scripts/scheduling_example.py

The cache and scheduler are the solution to exercise 3. The traces are this
page's own toy examples. The hit rate is hit tokens divided by prompt tokens,
which is how Section 6 defines the cache hit rate.
"""
import importlib.util
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sched_solution", HERE.parent / "exercises" / "3-cache-aware-scheduling" / "solution.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

prompts = {g: tuple(range(100 * g, 100 * g + 6)) for g in range(3)}
reqs = []
for j in range(3):
    for g in range(3):
        reqs.append(prompts[g] + (1000 + 10 * g + j, 2000 + 10 * g + j))

print("Toy trace: 3 shared prompts of 6 tokens, 3 requests each (2 own tokens), interleaved")
total = sum(map(len, reqs))
union = len({r[:k] for r in reqs for k in range(1, len(r) + 1)})
print(f"tokens in the requests: {total}; tokens in their union tree: {union}; best possible hits: {total - union}")
print(f"{'capacity':>9} {'fcfs hit rate':>14} {'lpm hit rate':>13}")
for cap in (10, 14, 20, 24, 30, 1000):
    h1, t1, o1 = mod.serve(reqs, cap, "fcfs")
    h2, t2, o2 = mod.serve(reqs, cap, "lpm")
    print(f"{cap:>9} {h1}/{t1} = {h1 / t1:.3f}   {h2}/{t2} = {h2 / t2:.3f}")
print("lpm order at capacity 20:", mod.serve(reqs, 20, "lpm")[2])
print("fcfs order at capacity 20:", mod.serve(reqs, 20, "fcfs")[2])

print()
print("Single batch: does longest-prefix-first equal depth-first (sorted) order?")
rng = random.Random(0)
trials, equal, lpm_better, fcfs_better = 400, 0, 0, 0
lpm_sum = fcfs_sum = tot_sum = 0
for _ in range(trials):
    base = [tuple(rng.randrange(3) for _ in range(rng.randrange(2, 5))) for _ in range(4)]
    rs = [rng.choice(base) + tuple(rng.randrange(4) for _ in range(rng.randrange(1, 5))) for _ in range(12)]
    cap = max(map(len, rs)) + rng.randrange(0, 12)
    lpm, tot, _ = mod.serve(rs, cap, "lpm")
    dfs = mod.serve(sorted(rs), cap, "fcfs")[0]
    fcfs = mod.serve(rs, cap, "fcfs")[0]
    equal += lpm == dfs
    lpm_better += lpm > fcfs
    fcfs_better += lpm < fcfs
    lpm_sum += lpm
    fcfs_sum += fcfs
    tot_sum += tot
print(f"{trials} random batches of 12 requests, cache = longest request plus 0 to 11 tokens")
print(f"lpm hits == dfs hits in {equal} of {trials}")
print(f"lpm beats fcfs in {lpm_better}, loses in {fcfs_better}")
print(f"mean hit rate: fcfs {fcfs_sum / tot_sum:.3f}, lpm {lpm_sum / tot_sum:.3f}")
