#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/whisper/scripts/check_mutants.py
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
    ("1-multitask-prompt", "task token before language token", [('"<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>"', '"<|startoftranscript|>", f"<|{task}|>", f"<|{language}|>"')]),
    ("1-multitask-prompt", "translate adds an English language token", [('f"<|{language}|>"', 'f"<|{language if task == \'transcribe\' else \'en\'}|>"')]),
    ("1-multitask-prompt", "<|notimestamps|> added when timestamps are on", [("if not timestamps:", "if timestamps:")]),
    ("1-multitask-prompt", "previous text placed after the start token", [('out += ["<|startofprev|>"] + list(prev_tokens)\n    out += ["<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>"]', 'out += ["<|startoftranscript|>", f"<|{language}|>", f"<|{task}|>"]\n    out += ["<|startofprev|>"] + list(prev_tokens)')]),
    ("1-multitask-prompt", "empty previous text still adds the marker", [("if prev_tokens:", "if prev_tokens is not None:")]),
    ("1-multitask-prompt", "unknown task accepted", [('if task not in ("transcribe", "translate"):', "if False:")]),
    ("1-multitask-prompt", "loss mask hides the whole prompt", [("for i in range(a + 1, b):", "for i in range(a + 1, len(tokens)):")]),
    ("1-multitask-prompt", "loss mask hides the marker tokens too", [("for i in range(a + 1, b):", "for i in range(a, b + 1):")]),
    ("2-timestamp-tokens", "truncate instead of round", [("int(round(seconds / 0.02))", "int(seconds / 0.02)")]),
    ("2-timestamp-tokens", "10 ms grid instead of 20 ms", [("int(round(seconds / 0.02))", "int(round(seconds * 100))"), ("k * 0.02", "k * 0.01")]),
    ("2-timestamp-tokens", "float noise from an unrounded format", [('f"<|{k * 0.02:.2f}|>"', 'f"<|{k * 0.02}|>"')]),
    ("2-timestamp-tokens", "end timestamp missing", [("+ list(toks) + [timestamp_token(end)]", "+ list(toks)")]),
    ("2-timestamp-tokens", "timestamps added when they are off", [("if timestamps:", "if True:")]),
    ("2-timestamp-tokens", "end-of-transcript token missing", [('return out + ["<|endoftranscript|>"]', "return out")]),
    ("3-log-mel", "last frame not dropped (3001 frames)", [("[:-1]  # drop the last frame", "  # keep all frames")]),
    ("3-log-mel", "no center padding (2998 frames)", [("padded = np.pad(audio, N_FFT // 2, mode=\"reflect\")", "padded = audio")]),
    ("3-log-mel", "zero padding instead of reflect", [('mode="reflect"', 'mode="constant"')]),
    ("3-log-mel", "natural log instead of log10", [("np.log10(", "np.log(")]),
    ("3-log-mel", "magnitude instead of power", [("np.abs(spec) ** 2", "np.abs(spec)")]),
    ("3-log-mel", "no dynamic-range clamp", [("    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)\n", "")]),
    ("3-log-mel", "floor of 1e-5 instead of 1e-10", [("np.maximum(mel, 1e-10)", "np.maximum(mel, 1e-5)")]),
    ("3-log-mel", "symmetric Hann window", [("np.hanning(N_FFT + 1)[:-1]", "np.hanning(N_FFT)")]),
    ("3-log-mel", "clamp taken per frame instead of globally", [("log_spec.max() - 8.0", "log_spec.max(axis=0, keepdims=True) - 8.0")]),
    ("4-decoding-fallback", "ratio computed over characters, not bytes", [("b = text.encode(\"utf-8\")", "b = text")]),
    ("4-decoding-fallback", "ratio inverted", [("return len(b) / len(zlib.compress(b))", "return len(zlib.compress(b)) / len(b)")]),
    ("4-decoding-fallback", "'and' instead of 'or'", [("> ratio_threshold or avg_logprob", "> ratio_threshold and avg_logprob")]),
    ("4-decoding-fallback", "non-strict ratio comparison", [("compression_ratio(text) > ratio_threshold", "compression_ratio(text) >= ratio_threshold")]),
    ("4-decoding-fallback", "non-strict log-probability comparison", [("avg_logprob < logprob_threshold", "avg_logprob <= logprob_threshold")]),
    ("4-decoding-fallback", "defaults ignored", [("ratio_threshold=2.4, logprob_threshold=-1.0):\n    \"\"\"Return True when the decode looks bad: too repetitive or too unlikely.\"\"\"\n    return compression_ratio(text) > ratio_threshold or avg_logprob < logprob_threshold", "ratio_threshold=2.4, logprob_threshold=-1.0):\n    \"\"\"Return True when the decode looks bad: too repetitive or too unlikely.\"\"\"\n    return compression_ratio(text) > 2.4 or avg_logprob < -1.0")]),
    ("4-decoding-fallback", "schedule stops before 1.0 (arange)", [("n = int(round((stop - start) / step))\n    return [round(start + i * step, 10) for i in range(n + 1)]", "n = int((stop - start) / step)\n    return [round(start + i * step, 10) for i in range(n)]")]),
    ("4-decoding-fallback", "schedule keeps float noise", [("round(start + i * step, 10)", "start + i * step")]),
    ("4-decoding-fallback", "returns None when every temperature fails", [("            return result\n    return result\n", "            return result\n    return None\n")]),
    ("4-decoding-fallback", "keeps going after a good decode", [("            return result\n", "            pass\n")]),
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
    survivors = 0
    ran = 0
    for folder, name, edits in MUTANTS:
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
