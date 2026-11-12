"""Toy timing model for the softmax-GEMM overlap of Section 3.1 and 3.2.

A task has a resource, a duration, and dependencies. A resource runs its tasks
one at a time, in issue order. The Tensor Core is one resource and the
special-function unit that runs exp is another, so a GEMM and a softmax can
overlap only when they come from different dependency chains.

Run: python3 scripts/pingpong_schedule.py
"""


def simulate(tasks):
    """tasks: list of (name, resource, duration, deps) in issue order.

    Returns (makespan, busy), where busy maps each resource to its busy time.
    """
    free, end, busy = {}, [], {}
    for name, res, dur, deps in tasks:
        start = max([free.get(res, 0.0)] + [end[d] for d in deps])
        end.append(start + dur)
        free[res] = start + dur
        busy[res] = busy.get(res, 0.0) + dur
    return max(end), busy


def serial_tasks(n, g, s):
    """One warpgroup, Algorithm 1: GEMM, then softmax, then the next GEMM."""
    tasks = []
    for j in range(n):
        prev = [len(tasks) - 1] if tasks else []
        tasks.append((f"G{j}", "tensor", g, prev))
        tasks.append((f"S{j}", "sfu", s, [len(tasks) - 1]))
    return tasks


def pingpong_tasks(n, g, s):
    """Two warpgroups. Barriers make the Tensor Core queue alternate A, B."""
    tasks, last_s = [], {"A": None, "B": None}
    for j in range(n):
        for wg in ("A", "B"):
            deps = [] if last_s[wg] is None else [last_s[wg]]
            tasks.append((f"{wg}.G{j}", "tensor", g, deps))
            tasks.append((f"{wg}.S{j}", "sfu", s, [len(tasks) - 1]))
            last_s[wg] = len(tasks) - 1
    return tasks


def two_stage_tasks(n, g, s):
    """One warpgroup, Algorithm 2: GEMM0 of block j+1 issues before GEMM1 of
    block j, so softmax of block j+1 overlaps GEMM1 of block j."""
    tasks, g0, sm, g1 = [], {}, {}, {}

    def add(key, store, res, dur, deps):
        tasks.append((key, res, dur, deps))
        store[key] = len(tasks) - 1

    add("G0.0", g0, "tensor", g / 2, [])
    add("S0", sm, "sfu", s, [g0["G0.0"]])
    for j in range(n):
        if j + 1 < n:
            add(f"G0.{j + 1}", g0, "tensor", g / 2, [])
        add(f"G1.{j}", g1, "tensor", g / 2, [sm[f"S{j}"]])
        if j + 1 < n:
            add(f"S{j + 1}", sm, "sfu", s, [g0[f"G0.{j + 1}"]])
    return tasks


def report(label, tasks):
    makespan, busy = simulate(tasks)
    print(f"  {label:<10} makespan {makespan:7.2f}   Tensor Core busy {busy['tensor']:6.2f}"
          f"   utilization {busy['tensor'] / makespan:6.1%}")


if __name__ == "__main__":
    print("The paper's arithmetic for head dimension 128 (Section 3.1, pingpong)")
    flops_per_exp = 4 * 128
    rate_ratio = 989 / 3.9
    print(f"  matmul FLOPs per exp = 4 * 128 = {flops_per_exp}")
    print(f"  matmul rate / exp rate = 989 / 3.9 = {rate_ratio:.1f}")
    print(f"  exp time / matmul time = {rate_ratio / flops_per_exp:.2f}")
    print()
    n = 16
    for label, g, s in (("FP16-like: softmax = 0.5 x GEMM", 2.0, 1.0),
                        ("FP8-like:  softmax = 1.0 x GEMM", 1.0, 1.0)):
        print(f"{label}  (g={g}, s={s}, n={n} blocks)")
        report("serial", serial_tasks(n, g, s))
        report("2-stage", two_stage_tasks(n, g, s))
        report("pingpong", pingpong_tasks(n, g, s))
        print()
    print("Register cost of the 2-stage pipeline (Section 3.2): an extra S_next of Br x Bc floats")
    for Br, Bc in ((128, 128),):
        print(f"  Br={Br}, Bc={Bc}: {Br * Bc * 4} bytes = {Br * Bc * 4 / 1024:.0f} KiB")
    print()
    print("Table 2 of the paper (non-causal FP16, batch 4, seqlen 8448, 16 heads, hdim 128)")
    full, no_pipe, no_ws = 661, 582, 570
    print(f"  full / no GEMM-softmax pipelining = {full} / {no_pipe} = {full / no_pipe:.3f}")
    print(f"  full / no warp specialization     = {full} / {no_ws} = {full / no_ws:.3f}")
