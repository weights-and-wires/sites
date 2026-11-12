#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/f5tts/scripts/check_mutants.py
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
HARNESS = ROOT / "static/runner/harness.py"
EXERCISES = pathlib.Path(__file__).resolve().parents[1] / "exercises"

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-infilling-target", "x_t uses 1 - t on the data", [("(1 - t) * x0 + t * x1", "t * x0 + (1 - t) * x1")]),
    ("1-infilling-target", "target is x0 - x1", [("target = x1 - x0", "target = x0 - x1")]),
    ("1-infilling-target", "cond keeps the masked frames", [("cond = (1 - m) * x1", "cond = m * x1")]),
    ("1-infilling-target", "cond ignores the mask", [("cond = (1 - m) * x1", "cond = x1")]),
    ("1-infilling-target", "loss over all frames", [("err = (pred - target) ** 2 * m", "err = (pred - target) ** 2 + 0 * m")]),
    ("1-infilling-target", "loss divides by all frames", [("(m.sum() * pred.shape[1])", "(len(m) * pred.shape[1])")]),
    ("1-infilling-target", "loss divides by masked frames only, not channels", [("(m.sum() * pred.shape[1])", "m.sum()")]),
    ("1-infilling-target", "loss on the context frames", [("* m\n", "* (1 - m)\n")]),
    ("2-filler-text", "one filler too few", [("[filler] * (n_frames - len(chars))", "[filler] * (n_frames - len(chars) - 1)")]),
    ("2-filler-text", "fillers first", [("list(chars) + [filler] * (n_frames - len(chars))", "[filler] * (n_frames - len(chars)) + list(chars)")]),
    ("2-filler-text", "no error when text is too long", [("if len(chars) > n_frames:", "if False:")]),
    ("2-filler-text", "duration rounds instead of flooring", [("int(ref_frames * len(gen_text) / len(ref_text))", "round(ref_frames * len(gen_text) / len(ref_text))")]),
    ("2-filler-text", "ratio inverted", [("len(gen_text) / len(ref_text)", "len(ref_text) / len(gen_text)")]),
    ("2-filler-text", "total omits the reference frames", [("return ref_frames + gen_frames", "return gen_frames")]),
    ("3-sway-sampling", "sign of s flipped", [("u + s * (np.cos", "u - s * (np.cos")]),
    ("3-sway-sampling", "cos argument without pi/2", [("np.cos(np.pi / 2 * u)", "np.cos(np.pi * u)")]),
    ("3-sway-sampling", "cos term minus 1 dropped", [("(np.cos(np.pi / 2 * u) - 1 + u)", "(np.cos(np.pi / 2 * u) + u)")]),
    ("3-sway-sampling", "upper bound written as 2/pi - 1", [("s_max = 2 / (np.pi - 2)", "s_max = 2 / np.pi - 1")]),
    ("3-sway-sampling", "no range check", [("if s < -1 - 1e-12 or s > s_max + 1e-12:", "if False:")]),
    ("3-sway-sampling", "steps sampled with nfe points", [("np.linspace(0, 1, nfe + 1)", "np.linspace(0, 1, nfe)")]),
    ("4-euler-cfg", "guidance weight on the wrong sign", [("v_cond + w * (v_cond - v_uncond)", "v_cond - w * (v_cond - v_uncond)")]),
    ("4-euler-cfg", "guidance as a convex mix", [("v_cond + w * (v_cond - v_uncond)", "(1 - w) * v_cond + w * v_uncond")]),
    ("4-euler-cfg", "constant step size", [("x = x + (b - a) * v", "x = x + (1.0 / (len(ts) - 1)) * v")]),
    ("4-euler-cfg", "velocity at the right end", [("vc = v_fn(x, a, True)", "vc = v_fn(x, b, True)"), ("v_fn(x, a, False)", "v_fn(x, b, False)")]),
    ("4-euler-cfg", "unconditional branch always called", [("if w == 0:", "if False:")]),
    ("4-euler-cfg", "conditional branch flag swapped", [("vc = v_fn(x, a, True)", "vc = v_fn(x, a, False)")]),
    ("4-euler-cfg", "x0 modified in place", [("x = np.array(x0, dtype=float)", "x = x0"), ("x = x + (b - a) * v", "x += (b - a) * v")]),
]


def run_tests(ex, source):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "submission.py").write_text(source)
        shutil.copy(ex / "tests.py", tmp / "tests.py")
        shutil.copy(HARNESS, tmp / "harness.py")
        proc = subprocess.run([sys.executable, "harness.py", "."], cwd=tmp, capture_output=True, text=True, timeout=120)
        lines = proc.stdout.strip().splitlines()
        if not lines:
            return None, proc.stderr[-400:]
        return json.loads(lines[-1]), ""


def main():
    skip_torch = False
    survivors = 0
    ran = 0
    for entry in MUTANTS:
        folder, name, edits = entry[:3]
        if len(entry) == 4 and skip_torch:
            print(f"SKIP  {folder}: {name} (torch not available)")
            continue
        ex = EXERCISES / folder
        solution = (ex / "solution.py").read_text()
        base, err = run_tests(ex, solution)
        if base is None or not all(r["ok"] for r in base):
            print(f"ERROR {folder}: the unmodified solution does not pass: {err}")
            survivors += 1
            continue
        mutated = solution
        for old, new in edits:
            if old not in mutated:
                print(f"ERROR {folder}: {name}: pattern not found: {old!r}")
                sys.exit(2)
            mutated = mutated.replace(old, new)
        results, err = run_tests(ex, mutated)
        ran += 1
        if results is None:
            print(f"ERROR {folder}: {name}: harness crashed: {err}")
            survivors += 1
            continue
        failed = [r["name"] for r in results if not r["ok"]]
        if failed:
            print(f"CAUGHT   {folder}: {name}\n         failing: {', '.join(failed)}")
        else:
            print(f"SURVIVED {folder}: {name}\n         no test failed")
            survivors += 1
    print(f"\n{ran - survivors} of {ran} mutants caught, {survivors} survived")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
