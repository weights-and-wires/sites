def simulate(tasks):
    """tasks: list of (name, resource, duration, deps) in issue order.

    A resource runs one task at a time, in issue order. A task starts when its
    resource is free and all of its dependencies (indices of earlier tasks)
    have ended. Returns (makespan, busy), with busy[resource] = total duration.
    """
    free, end, busy = {}, [], {}
    for name, resource, duration, deps in tasks:
        start = max([free.get(resource, 0.0)] + [end[d] for d in deps])
        end.append(start + duration)
        free[resource] = start + duration
        busy[resource] = busy.get(resource, 0.0) + duration
    return max(end), busy


def utilization(tasks, resource="tensor"):
    """Fraction of the makespan during which `resource` is busy."""
    makespan, busy = simulate(tasks)
    return busy[resource] / makespan


def serial_tasks(n, g, s):
    """One warpgroup (Algorithm 1): GEMM, then softmax, then the next GEMM."""
    tasks = []
    for j in range(n):
        prev = [len(tasks) - 1] if tasks else []
        tasks.append((f"G{j}", "tensor", g, prev))
        tasks.append((f"S{j}", "sfu", s, [len(tasks) - 1]))
    return tasks


def pingpong_tasks(n, g, s):
    """Two warpgroups A and B; A issues its GEMM before B in every round."""
    tasks, last_s = [], {"A": None, "B": None}
    for j in range(n):
        for wg in ("A", "B"):
            deps = [] if last_s[wg] is None else [last_s[wg]]
            tasks.append((f"{wg}.G{j}", "tensor", g, deps))
            tasks.append((f"{wg}.S{j}", "sfu", s, [len(tasks) - 1]))
            last_s[wg] = len(tasks) - 1
    return tasks
