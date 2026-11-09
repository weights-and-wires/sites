#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/llama/scripts/check_mutants.py
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
    ("1-rmsnorm", "subtract the mean (LayerNorm without bias)",
     [("ms = np.mean(x**2, axis=-1, keepdims=True)", "x = x - x.mean(axis=-1, keepdims=True)\n    ms = np.mean(x**2, axis=-1, keepdims=True)")]),
    ("1-rmsnorm", "mean over the first axis", [("np.mean(x**2, axis=-1, keepdims=True)", "np.mean(x**2, axis=0, keepdims=True)")]),
    ("1-rmsnorm", "no square root (divide by the mean square)", [("np.sqrt(ms + eps)", "(ms + eps)")]),
    ("1-rmsnorm", "eps added after the square root", [("np.sqrt(ms + eps)", "(np.sqrt(ms) + eps)")]),
    ("1-rmsnorm", "sum instead of mean", [("np.mean(x**2, axis=-1, keepdims=True)", "np.sum(x**2, axis=-1, keepdims=True)")]),
    ("1-rmsnorm", "gain ignored", [(" * g", "")]),
    ("2-swiglu", "SiLU on the value branch", [("silu(x @ W1) * (x @ V)", "(x @ W1) * silu(x @ V)")]),
    ("2-swiglu", "no gate nonlinearity (bilinear GLU)", [("silu(x @ W1) * (x @ V)", "(x @ W1) * (x @ V)")]),
    ("2-swiglu", "sigmoid instead of SiLU", [("return a / (1.0 + np.exp(-a))", "return 1.0 / (1.0 + np.exp(-a))")]),
    ("2-swiglu", "add instead of multiply the branches", [("silu(x @ W1) * (x @ V)", "silu(x @ W1) + (x @ V)")]),
    ("2-swiglu", "hidden width 4d (forget the 2/3)", [("int(2 * 4 * d / 3)", "int(4 * d)")]),
    ("2-swiglu", "round to nearest multiple instead of up", [("math.ceil(int(2 * 4 * d / 3) / multiple_of)", "round(int(2 * 4 * d / 3) / multiple_of)")]),
    ("3-rope", "split-half pairing instead of adjacent pairs", [
        ("out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s", "out[:, : d // 2] = x[:, : d // 2] * c - x[:, d // 2 :] * s"),
        ("out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c", "out[:, d // 2 :] = x[:, : d // 2] * s + x[:, d // 2 :] * c")]),
    ("3-rope", "clockwise rotation", [("out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s", "out[:, 0::2] = x[:, 0::2] * c + x[:, 1::2] * s"),
                                       ("out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c", "out[:, 1::2] = -x[:, 0::2] * s + x[:, 1::2] * c")]),
    ("3-rope", "exponent -j/d instead of -2j/d", [("-2.0 * np.arange(d // 2) / d", "-1.0 * np.arange(d // 2) / d")]),
    ("3-rope", "base hard-coded to 10000", [("theta = base ** (", "theta = 10000.0 ** (")]),
    ("3-rope", "every row rotated by position 1", [("np.asarray(positions, dtype=float)[:, None]", "np.ones((n, 1))")]),
    ("4-llama-block", "post-norm instead of pre-norm", [
        ('h = rmsnorm(x, p["g1"])', "h = x"),
        ("x = x + attn", 'x = rmsnorm(x + attn, p["g1"])'),
        ('h2 = rmsnorm(x, p["g2"])\n    return x + swiglu_ffn(h2, p["W1"], p["V"], p["W2"])', 'return rmsnorm(x + swiglu_ffn(x, p["W1"], p["V"], p["W2"]), p["g2"])')]),
    ("4-llama-block", "no causal mask", [("np.where(np.tril(np.ones((n, n), dtype=bool)), scores, -1e9)", "scores")]),
    ("4-llama-block", "no RoPE", [("q = np.stack([apply_rope(q[i], positions) for i in range(n_heads)])", ""),
                                  ("k = np.stack([apply_rope(k[i], positions) for i in range(n_heads)])", "")]),
    ("4-llama-block", "RoPE applied to the values too", [("heads = softmax(scores, axis=-1) @ v", "v = np.stack([apply_rope(v[i], positions) for i in range(n_heads)])\n    heads = softmax(scores, axis=-1) @ v")]),
    ("4-llama-block", "RoPE applied to the input before the projections", [
        ('h = rmsnorm(x, p["g1"])', 'h = rmsnorm(x, p["g1"])\n    hp = apply_rope(h, positions)'),
        ('split(h @ p["Wq"]), split(h @ p["Wk"])', 'split(hp @ p["Wq"]), split(hp @ p["Wk"])'),
        ("q = np.stack([apply_rope(q[i], positions) for i in range(n_heads)])", ""),
        ("k = np.stack([apply_rope(k[i], positions) for i in range(n_heads)])", "")]),
    ("4-llama-block", "no residual around attention", [("x = x + attn", "x = attn")]),
    ("4-llama-block", "no residual around the FFN", [('return x + swiglu_ffn(h2,', "return swiglu_ffn(h2,")]),
    ("4-llama-block", "scale by sqrt(d) instead of sqrt(d_head)", [("/ np.sqrt(d_head)", "/ np.sqrt(d)")]),
    ("4-llama-block", "heads reshaped without the transpose", [("return M.reshape(n, n_heads, d_head).transpose(1, 0, 2)", "return M.reshape(n_heads, n, d_head)")]),
    ("4-llama-block", "heads merged without transposing back", [("heads.transpose(1, 0, 2).reshape(n, d)", "heads.reshape(n, d)")]),
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
