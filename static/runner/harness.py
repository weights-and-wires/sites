"""Run every test_* function in tests.py against submission.py.

The browser runner (Pyodide) and scripts/check_exercises.py both use this
file, so a test behaves the same on the page and in CI. run() returns a list
of {"name", "ok", "message"} dictionaries in definition order.
"""

import importlib
import sys
import traceback


def _failure(exc):
    """Describe a failed test: the assert message, or the failing line."""
    if isinstance(exc, AssertionError) and str(exc):
        return str(exc)
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename.endswith(("tests.py", "submission.py"))]
    where = ""
    if frames:
        f = frames[-1]
        where = f" at {f.filename.rsplit('/', 1)[-1]} line {f.lineno}: {f.line}"
    if isinstance(exc, AssertionError):
        return "assertion failed" + where
    if isinstance(exc, NotImplementedError):
        return "not implemented yet" + where
    return f"{type(exc).__name__}: {exc}{where}"


def run():
    for name in ("tests", "submission"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        tests = importlib.import_module("tests")
    except BaseException as exc:  # a syntax error in the submission lands here
        return [{"name": "import", "ok": False, "message": _failure(exc)}]
    results = []
    for name, fn in list(vars(tests).items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            results.append({"name": name, "ok": True, "message": ""})
        except BaseException as exc:
            results.append({"name": name, "ok": False, "message": _failure(exc)})
    if not results:
        results.append({"name": "tests", "ok": False, "message": "tests.py defines no test_ functions"})
    return results


if __name__ == "__main__":
    import json

    sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
    print(json.dumps(run()))
