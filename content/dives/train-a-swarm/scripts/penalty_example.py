"""Worked example A: the latency penalty from the post, with our numbers."""

beta, pbar, tmax, task = 0.5, 0.5, 100.0, 0.8
for name, t in [("single agent", 100.0), ("swarm", 46.0)]:
    r = task - beta * pbar * t / tmax
    print(f"{name:13s} T={t:5.1f} min  R = {task} - {beta} * {pbar} * {t / tmax:.2f} = {r:.3f}")
print("gap:", round((task - beta * pbar * 46 / tmax) - (task - beta * pbar * 100 / tmax), 3))
print("without the penalty both score", task)
