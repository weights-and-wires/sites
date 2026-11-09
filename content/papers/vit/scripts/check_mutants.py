#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/vit/scripts/check_mutants.py
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
    ("1-patchify", "reshape to (N, P*P*C) without the transpose", [
        ("grid = img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4)\n    return grid.reshape((H // P) * (W // P), P * P * C)",
         "return img.reshape((H // P) * (W // P), P * P * C)")]),
    ("1-patchify", "patches in column-major order", [
        ("img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4)", "img.reshape(H // P, P, W // P, P, C).transpose(2, 0, 1, 3, 4)")]),
    ("1-patchify", "channels-first flatten inside a patch", [
        ("img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4)", "img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 4, 1, 3)")]),
    ("1-patchify", "unpatchify skips the transpose", [
        ("patches.reshape(H // P, W // P, P, P, C).transpose(0, 2, 1, 3, 4)", "patches.reshape(H // P, W // P, P, P, C)")]),
    ("2-embed-patches", "class token appended at the end", [
        ("np.vstack([x_class[None, :], proj])", "np.vstack([proj, x_class[None, :]])")]),
    ("2-embed-patches", "position embedding skips the class token", [
        ("return np.vstack([x_class[None, :], proj]) + E_pos", "z = np.vstack([x_class[None, :], proj])\n    z[1:] += E_pos[1:]\n    return z")]),
    ("2-embed-patches", "position embeddings forgotten", [
        ("np.vstack([x_class[None, :], proj]) + E_pos", "np.vstack([x_class[None, :], proj])")]),
    ("2-embed-patches", "patches taken in the wrong order", [
        ("patchify(img, P) @ E", "patchify(img, P)[::-1] @ E")]),
    ("3-encoder-readout", "post-LN instead of pre-LN", [
        ("z = z + msa(layernorm(z, p[\"g1\"], p[\"b1\"]), p)\n    return z + mlp(layernorm(z, p[\"g2\"], p[\"b2\"]), p)",
         "z = layernorm(z + msa(z, p), p[\"g1\"], p[\"b1\"])\n    return layernorm(z + mlp(z, p), p[\"g2\"], p[\"b2\"])")]),
    ("3-encoder-readout", "residual around the MLP added to the normalized input", [
        ("return z + mlp(layernorm(z, p[\"g2\"], p[\"b2\"]), p)", "n = layernorm(z, p[\"g2\"], p[\"b2\"])\n    return n + mlp(n, p)")]),
    ("3-encoder-readout", "no residual around the attention", [
        ("z = z + msa(layernorm(z, p[\"g1\"], p[\"b1\"]), p)", "z = msa(layernorm(z, p[\"g1\"], p[\"b1\"]), p)")]),
    ("3-encoder-readout", "final LayerNorm skipped", [
        ("return layernorm(z[0], g, b)", "return z[0]")]),
    ("3-encoder-readout", "mean of all tokens instead of the class token", [
        ("return layernorm(z[0], g, b)", "return layernorm(z.mean(0), g, b)")]),
    ("4-resize-position-embeddings", "align_corners=False coordinates", [
        ("pos = np.linspace(0, g - 1, g_new) if g_new > 1 else np.zeros(1)",
         "pos = np.clip((np.arange(g_new) + 0.5) * g / g_new - 0.5, 0, g - 1)")]),
    ("4-resize-position-embeddings", "nearest neighbour instead of bilinear", [
        ("return i0, i0 + 1, pos - i0", "return i0, i0 + 1, np.round(pos - i0)")]),
    ("4-resize-position-embeddings", "class row interpolated away", [
        ("return np.vstack([E_pos[:1], out.reshape(g_new * g_new, D)])", "return np.vstack([out.reshape(g_new * g_new, D)[:1] * 0 + out.mean(), out.reshape(g_new * g_new, D)])")]),
    ("4-resize-position-embeddings", "grid read in column-major order", [
        ("grid = E_pos[1:].reshape(g, g, D)", "grid = E_pos[1:].reshape(g, g, D).transpose(1, 0, 2)")]),
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
    for entry in MUTANTS:
        folder, name, edits = entry
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
