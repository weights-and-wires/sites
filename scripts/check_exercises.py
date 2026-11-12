#!/usr/bin/env python3
"""Check paper exercises: the starter fails, the solution passes, both fast.

For every exercise folder (content/papers/<id>/exercises/<n>-<slug>/), this
script runs tests.py twice through static/runner/harness.py, the same harness
the browser runner uses:

1. With starter.py as submission.py. At least one test must fail.
2. With solution.py as submission.py. Every test must pass.

Each run must finish within the time limit: 5 seconds for NumPy exercises
(runtime "pyodide") and 60 seconds for PyTorch exercises (runtime "torch"), because CI runners are several times slower than a laptop.
PyTorch exercises are skipped when torch isn't installed, unless you pass
--require-torch.

Usage:
    python3 scripts/check_exercises.py [PATH ...] [--require-torch]

PATH is a paper folder, an exercise folder, a dive folder, or content/papers
and content/dives (the default).

It also runs every deep-dive cell (content/dives/<id>/cells/*.py): each must
import only modules the browser runner has, exit without an error, and finish
within 5 seconds.
"""

import argparse
import ast
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
HARNESS = ROOT / "static/runner/harness.py"
LIMITS = {"pyodide": 5.0, "torch": 60.0}
# Modules that Pyodide's core plus NumPy provides.
PYODIDE_OK = {
    "numpy", "math", "cmath", "itertools", "functools", "collections", "random", "typing",
    "dataclasses", "heapq", "bisect", "re", "string", "json", "fractions", "statistics",
    "decimal", "operator", "copy", "struct", "zlib", "hashlib", "time", "textwrap",
    "submission", "__future__", "enum", "abc", "contextlib", "warnings", "sys",
}


def exercise_dirs(paths):
    for p in paths:
        p = pathlib.Path(p).resolve()
        if (p / "meta.json").exists():
            yield p
        else:
            yield from sorted(d.parent for d in p.rglob("exercises/*/meta.json"))


def cell_files(paths):
    for p in paths:
        p = pathlib.Path(p).resolve()
        if p.suffix == ".py" and p.parent.name == "cells":
            yield p
        else:
            yield from sorted(p.rglob("cells/*.py"))


def check_cell(cell):
    name = f"{cell.parent.parent.name}/cells/{cell.name}"
    bad = imports(cell.read_text()) - PYODIDE_OK
    if bad:
        return name, "fail", ["-", f"imports {sorted(bad)}, which the browser runner can't load"]
    limit = LIMITS["pyodide"]
    with tempfile.TemporaryDirectory() as tmp:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(cell)], cwd=tmp, capture_output=True, text=True, timeout=limit + 30
            )
        except subprocess.TimeoutExpired:
            return name, "fail", ["-", "timed out"]
        elapsed = time.monotonic() - start
    problems = []
    if proc.returncode != 0:
        problems.append("the cell raised: " + (proc.stderr or proc.stdout).strip()[-800:])
    if elapsed > limit:
        problems.append(f"the cell took {elapsed:.1f} s; the limit is {limit:.0f} s")
    return name, "fail" if problems else "ok", [f"{elapsed:.2f} s"] + problems


def imports(src):
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def run(ex, submission, limit):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        shutil.copy(ex / submission, tmp / "submission.py")
        shutil.copy(ex / "tests.py", tmp / "tests.py")
        shutil.copy(HARNESS, tmp / "harness.py")
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "harness.py", "."], cwd=tmp, capture_output=True, text=True, timeout=limit + 30
            )
        except subprocess.TimeoutExpired:
            return None, limit + 30, "timed out"
        elapsed = time.monotonic() - start
        lines = proc.stdout.strip().splitlines()
        if proc.returncode != 0 or not lines:
            return None, elapsed, (proc.stderr or proc.stdout).strip()[-800:]
        return json.loads(lines[-1]), elapsed, ""


def check(ex, require_torch):
    meta = json.loads((ex / "meta.json").read_text())
    runtime = meta.get("runtime", "pyodide")
    name = f"{ex.parent.parent.name}/{ex.name}"
    problems = []
    for f in ("starter.py", "tests.py", "solution.py"):
        if not (ex / f).exists():
            return name, "fail", [f"missing {f}"]
    tests_src = (ex / "tests.py").read_text()
    if "from submission import" not in tests_src and "import submission" not in tests_src:
        problems.append("tests.py must import from submission")
    if runtime == "pyodide":
        for f in ("starter.py", "tests.py", "solution.py"):
            bad = imports((ex / f).read_text()) - PYODIDE_OK
            if bad:
                problems.append(f"{f} imports {sorted(bad)}, which the browser runner can't load")
    elif runtime == "torch" and importlib.util.find_spec("torch") is None:
        if require_torch:
            return name, "fail", ["torch is not installed"]
        return name, "skip", ["torch is not installed"]
    limit = LIMITS[runtime]
    starter, t1, err1 = run(ex, "starter.py", limit)
    solution, t2, err2 = run(ex, "solution.py", limit)
    if starter is None:
        problems.append(f"starter crashed the harness: {err1}")
    elif all(r["ok"] for r in starter):
        problems.append("the starter passes every test; it must fail at least one")
    if solution is None:
        problems.append(f"solution crashed the harness: {err2}")
    else:
        failed = [f"{r['name']}: {r['message']}" for r in solution if not r["ok"]]
        if failed:
            problems.append("the solution fails: " + "; ".join(failed))
    for label, t in (("starter", t1), ("solution", t2)):
        if t > limit:
            problems.append(f"{label} took {t:.1f} s; the limit is {limit:.0f} s")
    detail = [f"starter {t1:.2f} s, solution {t2:.2f} s"] + problems
    return name, "fail" if problems else "ok", detail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[str(ROOT / "content/papers"), str(ROOT / "content/dives")])
    ap.add_argument("--require-torch", action="store_true")
    args = ap.parse_args()
    counts = {"ok": 0, "fail": 0, "skip": 0}
    paths = [p for p in args.paths if pathlib.Path(p).exists()]
    jobs = [lambda ex=ex: check(ex, args.require_torch) for ex in exercise_dirs(paths)]
    jobs += [lambda c=c: check_cell(c) for c in cell_files(paths)]
    for job in jobs:
        name, status, detail = job()
        counts[status] += 1
        print(f"{status.upper():4} {name}: {detail[0]}")
        for d in detail[1:]:
            print(f"     {d}")
    print(f"\n{counts['ok']} passed, {counts['fail']} failed, {counts['skip']} skipped")
    sys.exit(1 if counts["fail"] else 0)


if __name__ == "__main__":
    main()
