def simulate(tasks):
    """tasks: list of (name, resource, duration, deps) in issue order.

    A resource runs one task at a time, in issue order. A task starts when its
    resource is free and all of its dependencies (indices of earlier tasks)
    have ended. Returns (makespan, busy), with busy[resource] = total duration.
    """
    raise NotImplementedError


def utilization(tasks, resource="tensor"):
    """Fraction of the makespan during which `resource` is busy."""
    raise NotImplementedError


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
    raise NotImplementedError
