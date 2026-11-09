#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/moshi/scripts/check_mutants.py
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
    ("1-stream-budget", "strides added instead of multiplied", [("math.prod(strides)", "sum(strides)")]),
    ("1-stream-budget", "bitrate counts entries, not bits", [("math.log2(codebook_size)", "codebook_size")]),
    ("1-stream-budget", "bitrate uses one codebook", [("frame_rate * n_codebooks * math.log2", "frame_rate * math.log2")]),
    ("1-stream-budget", "flattened and temporal swapped", [("frames * n_codebooks if flattened else frames", "frames if flattened else frames * n_codebooks")]),
    ("1-stream-budget", "frames not floored", [("frames = int(seconds * frame_rate)", "frames = seconds * frame_rate")]),
    ("1-stream-budget", "K = 2Q (forgets the text stream)", [("2 * n_codebooks + 1", "2 * n_codebooks")]),
    ("1-stream-budget", "K = Q + 1 (one audio stream)", [("2 * n_codebooks + 1", "n_codebooks + 1")]),
    ("2-acoustic-delay", "np.roll wraps codes to the front", [("out = np.zeros_like(codes)", "out = codes.copy()"), ("out[q, d:] = codes[q, : T - d]", "out[q] = np.roll(codes[q], d)")]),
    ("2-acoustic-delay", "shift left (advance) instead of delay", [("out[q, d:] = codes[q, : T - d]", "out[q, : T - d] = codes[q, d:]")]),
    ("2-acoustic-delay", "latency without the base frame", [("(max(delays) + 1) * frame_ms", "max(delays) * frame_ms")]),
    ("2-acoustic-delay", "latency sums the delays", [("(max(delays) + 1) * frame_ms", "(sum(delays) + 1) * frame_ms")]),
    ("2-acoustic-delay", "undo reads from the wrong offset", [("V[q, d : d + width]", "V[q, :width]")]),
    ("3-joint-sequence", "semantic row delayed too", [("delays = [0] + [tau] * (Q - 1)", "delays = [tau] * Q")]),
    ("3-joint-sequence", "text row delayed", [("np.asarray(W)[None]", "apply_delays(np.asarray(W)[None], [tau])")]),
    ("3-joint-sequence", "user rows before Moshi rows", [("apply_delays(A, delays), apply_delays(B, delays)", "apply_delays(B, delays), apply_delays(A, delays)")]),
    ("3-joint-sequence", "text row last", [("np.vstack([np.asarray(W)[None], apply_delays(A, delays), apply_delays(B, delays)])", "np.vstack([apply_delays(A, delays), apply_delays(B, delays), np.asarray(W)[None]])")]),
    ("3-joint-sequence", "sampled rows off by one (paper's 2..2+Q)", [("list(range(Q + 1))", "list(range(Q + 2))")]),
    ("3-joint-sequence", "sampled rows omit the text row", [("list(range(Q + 1))", "list(range(1, Q + 1))")]),
    ("4-text-alignment", "no EPAD", [("W[0] = EPAD", "pass"), ("elif W[t - 2] == PAD:\n            W[t - 2] = EPAD", "elif False:\n            pass")]),
    ("4-text-alignment", "EPAD overwrites the previous word", [("elif W[t - 2] == PAD:", "else:")]),
    ("4-text-alignment", "word at step 1 not shifted", [("            t = 2\n", "")]),
    ("4-text-alignment", "first token one slot late (Equation 5 read literally)", [("W[t - 1 + j] = tok", "W[t + j] = tok"), ("if t - 1 + j < T:", "if t + j < T:")]),
    ("4-text-alignment", "tokens run past T", [("            if t - 1 + j < T:\n                W[t - 1 + j] = tok", "            W.append(tok) if t - 1 + j >= T else W.__setitem__(t - 1 + j, tok)")]),
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
