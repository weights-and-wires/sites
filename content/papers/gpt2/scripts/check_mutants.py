#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/gpt2/scripts/check_mutants.py
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

PRE_LN = ('x = x + attention(layer_norm(x, p["ln1_g"], p["ln1_b"]), p, n_heads)\n'
          '    x = x + ffn(layer_norm(x, p["ln2_g"], p["ln2_b"]), p)')
POST_LN = ('x = layer_norm(x + attention(x, p, n_heads), p["ln1_g"], p["ln1_b"])\n'
           '    x = layer_norm(x + ffn(x, p), p["ln2_g"], p["ln2_b"])')

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-next-token-loss", "no shift: row t scored against tokens[t]",
     [("z = logits[:-1]\n    targets = tokens[1:]", "z = logits\n    targets = tokens")]),
    ("1-next-token-loss", "unstable log-softmax (no max subtraction)",
     [("shifted = z - z.max(axis=-1, keepdims=True)", "shifted = z")]),
    ("1-next-token-loss", "perplexity as 2 ** loss",
     [("np.exp(lm_loss(logits, tokens))", "2 ** lm_loss(logits, tokens)")]),
    ("1-next-token-loss", "mean over T instead of T - 1 predictions",
     [(".mean())\n\n\ndef perplexity", ".sum() / len(tokens))\n\n\ndef perplexity")]),
    ("2-pre-ln-gpt", "post-LN block", [(PRE_LN, POST_LN)]),
    ("2-pre-ln-gpt", "no final layer norm",
     [('    x = layer_norm(x, params["lnf_g"], params["lnf_b"])\n', "")]),
    ("2-pre-ln-gpt", "no position embeddings",
     [(' + params["wpe"][: len(tokens)]', "")]),
    ("2-pre-ln-gpt", "no residual around the feed-forward network",
     [('x = x + ffn(layer_norm(x, p["ln2_g"], p["ln2_b"]), p)', 'x = ffn(layer_norm(x, p["ln2_g"], p["ln2_b"]), p)')]),
    ("2-pre-ln-gpt", "residual scale counts blocks, not residual layers",
     [("1.0 / np.sqrt(2 * n_blocks)", "1.0 / np.sqrt(n_blocks)")]),
    ("2-pre-ln-gpt", "output projection is not the embedding matrix",
     [('return x @ params["wte"].T', 'return x @ params["wte"][::-1].T')]),
    ("3-temperature-top-k", "temperature multiplies the logits",
     [("/ temperature", "* temperature")]),
    ("3-temperature-top-k", "top-k keeps the k smallest logits",
     [("np.argsort(z)[-top_k:]", "np.argsort(z)[:top_k]")]),
    ("3-temperature-top-k", "masked logits set to 0 instead of -inf",
     [("np.full_like(z, -np.inf)", "np.zeros_like(z)")]),
    ("3-temperature-top-k", "top-k applied after softmax without renormalizing",
     [("        z = masked\n    e = np.exp(z - z.max())\n    return e / e.sum()",
       "        pass\n    e = np.exp(z - z.max())\n    p = e / e.sum()\n"
       "    if top_k is not None and top_k < len(z):\n        q = np.zeros_like(p)\n"
       "        q[keep] = p[keep]\n        return q\n    return p")]),
    ("4-byte-pretokenize", "a space stands alone instead of joining the word",
     [('if j < n and text[j - 1] == " ":', "if False:")]),
    ("4-byte-pretokenize", "letters, digits, and punctuation share one category",
     [('    if ch.isalpha():\n        return "letter"\n    if ch.isdigit():\n        return "digit"\n    return "other"',
       '    return "letter"')]),
    ("4-byte-pretokenize", "the whole whitespace run joins the next word",
     [("                if j - 1 > i:\n                    pieces.append(text[i:j - 1])\n"
       "                i = j - 1\n                j = _run_end(text, j)",
       "                j = _run_end(text, j)")]),
    ("4-byte-pretokenize", "a newline joins the next word like a space",
     [('text[j - 1] == " "', "text[j - 1].isspace()")]),
    ("4-byte-pretokenize", "bytes from code points instead of UTF-8",
     [('list(text.encode("utf-8"))', "[ord(c) for c in text]")]),
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
