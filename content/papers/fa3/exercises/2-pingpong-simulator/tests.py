from submission import pingpong_tasks, serial_tasks, simulate, utilization


def test_independent_resources_overlap():
    tasks = [("a", "tensor", 4.0, []), ("b", "sfu", 3.0, [])]
    makespan, busy = simulate(tasks)
    assert makespan == 4.0, f"expected makespan 4.0, got {makespan}"
    assert busy == {"tensor": 4.0, "sfu": 3.0}, f"expected busy {{'tensor': 4.0, 'sfu': 3.0}}, got {busy}"


def test_one_resource_runs_one_task_at_a_time():
    tasks = [("a", "tensor", 4.0, []), ("b", "tensor", 2.0, [])]
    makespan, _ = simulate(tasks)
    assert makespan == 6.0, f"expected makespan 6.0 (the tasks cannot overlap), got {makespan}"


def test_dependencies_delay_a_task():
    tasks = [("a", "tensor", 4.0, []), ("b", "sfu", 1.0, [0]), ("c", "tensor", 1.0, [1])]
    makespan, _ = simulate(tasks)
    assert makespan == 6.0, f"expected makespan 6.0 (a, then b, then c), got {makespan}"


def test_resource_queue_is_in_issue_order():
    # t2 has no dependencies but is issued after t1 on the same resource, so it waits for t1.
    tasks = [("t0", "tensor", 4.0, []), ("t1", "sfu", 1.0, [0]), ("t2", "sfu", 1.0, [])]
    makespan, _ = simulate(tasks)
    assert makespan == 6.0, f"expected makespan 6.0 (t2 runs after t1 ends at 5), got {makespan}"


def test_serial_schedule_matches_the_closed_form():
    n, g, s = 10, 2.0, 1.0
    makespan, _ = simulate(serial_tasks(n, g, s))
    assert makespan == n * (g + s), f"expected makespan {n * (g + s)}, got {makespan}"
    u = utilization(serial_tasks(n, g, s))
    assert abs(u - g / (g + s)) < 1e-12, f"expected utilization {g / (g + s):.4f}, got {u:.4f}"


def test_pingpong_hides_softmax_when_gemm_is_longer():
    n, g, s = 16, 2.0, 1.0
    makespan, busy = simulate(pingpong_tasks(n, g, s))
    assert makespan == 2 * n * g + s, f"expected makespan {2 * n * g + s}, got {makespan}"
    assert busy["tensor"] == 2 * n * g, f"expected Tensor Core busy {2 * n * g}, got {busy['tensor']}"
    assert busy["sfu"] == 2 * n * s, f"expected sfu busy {2 * n * s}, got {busy['sfu']}"
    u = utilization(pingpong_tasks(n, g, s))
    want = 2 * n * g / (2 * n * g + s)
    assert abs(u - want) < 1e-12, f"expected utilization {want:.4f}, got {u:.4f}"


def test_pingpong_fp8_like_timing():
    n = 16
    u = utilization(pingpong_tasks(n, 1.0, 1.0))
    assert abs(u - 32 / 33) < 1e-12, f"expected utilization {32 / 33:.4f}, got {u:.4f}"
    assert u > utilization(serial_tasks(n, 1.0, 1.0)) + 0.4, "expected pingpong to beat serial by a wide margin"


def test_pingpong_when_softmax_is_the_bottleneck():
    # The sfu now limits the schedule: it runs 2 * n * s = 24 after the first GEMM ends at g = 1.
    n, g, s = 4, 1.0, 3.0
    makespan, _ = simulate(pingpong_tasks(n, g, s))
    assert makespan == g + 2 * n * s, f"expected makespan {g + 2 * n * s}, got {makespan}"


def test_pingpong_task_count():
    tasks = pingpong_tasks(5, 2.0, 1.0)
    tensor = [t for t in tasks if t[1] == "tensor"]
    sfu = [t for t in tasks if t[1] == "sfu"]
    assert len(tensor) == 10 and len(sfu) == 10, f"expected 10 GEMMs and 10 softmaxes, got {len(tensor)} and {len(sfu)}"


def test_pingpong_issue_order_alternates_warpgroups():
    tensor = [t[0] for t in pingpong_tasks(3, 2.0, 1.0) if t[1] == "tensor"]
    want = ["A.G0", "B.G0", "A.G1", "B.G1", "A.G2", "B.G2"]
    assert tensor == want, f"expected Tensor Core issue order {want}, got {tensor}"


def test_pingpong_dependencies():
    tasks = pingpong_tasks(3, 2.0, 1.0)
    names = [t[0] for t in tasks]
    for name, _, _, deps in tasks:
        wg, kind = name.split(".")[0], name.split(".")[1]
        dep_names = sorted(names[d] for d in deps)
        if kind == "G0":
            want = []
        elif kind[0] == "G":
            want = [f"{wg}.S{int(kind[1:]) - 1}"]
        else:
            want = [f"{wg}.G{kind[1:]}"]
        assert dep_names == want, f"{name}: expected dependencies {want}, got {dep_names}"
