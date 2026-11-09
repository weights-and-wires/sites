#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/valle/scripts/check_mutants.py
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

TRI = "np.tril(np.ones((n_audio, n_audio), dtype=bool))"

# (exercise folder, mutant name, [(old, new), ...])
MUTANTS = [
    ("1-ar-mask", "acoustic rows cannot see the text", [("m[n_text:, :n_text] = True", "m[n_text:, :n_text] = False")]),
    ("1-ar-mask", "acoustic block is not causal (NAR-style)", [(TRI, "np.ones((n_audio, n_audio), dtype=bool)")]),
    ("1-ar-mask", "diagonal hidden in the acoustic block", [(TRI, "np.tril(np.ones((n_audio, n_audio), dtype=bool), k=-1)")]),
    ("1-ar-mask", "text rows see the acoustic tokens", [("m[:n_text, :n_text] = True", "m[:n_text, :] = True")]),
    ("1-ar-mask", "NAR mask is causal", [("return np.ones((n, n), dtype=bool)", "return np.tril(np.ones((n, n), dtype=bool))")]),
    ("2-nar-inputs", "stage i also sums codebook i (off by one)", [("for j in range(i - 1))", "for j in range(i))")]),
    ("2-nar-inputs", "prompt uses only the first codebook", [("tables[j][prompt_codes[:, j]] for j in range(K)", "tables[j][prompt_codes[:, j]] for j in range(1)")]),
    ("2-nar-inputs", "prompt placed after the target", [("[text_emb, e_prompt, e_target]", "[text_emb, e_target, e_prompt]")]),
    ("2-nar-inputs", "stage range not checked", [("if not 2 <= i <= K:", "if False:")]),
    ("2-nar-inputs", "AdaLN shifts before scaling", [("a * (h - mu) / np.sqrt(var + 1e-5) + b", "a * ((h - mu) / np.sqrt(var + 1e-5) + b)")]),
    ("3-prompt-posterior", "likelihoods multiplied instead of log-summed", [
        ("logp = np.log(np.asarray(prior, dtype=float)).copy()", "logp = np.asarray(prior, dtype=float).copy()"),
        ("logp += np.log(trans[:, a, b])", "logp *= trans[:, a, b]"),
        ("p = np.exp(logp - logp.max())", "p = logp"),
    ]),
    ("3-prompt-posterior", "transition indexed backwards", [("trans[:, a, b]", "trans[:, b, a]")]),
    ("3-prompt-posterior", "prior ignored", [("np.log(np.asarray(prior, dtype=float)).copy()", "np.zeros(len(prior))")]),
    ("3-prompt-posterior", "mixture weights are the prior, not the posterior", [("speaker_posterior(prefix, trans, prior) @", "np.asarray(prior) @")]),
    ("3-prompt-posterior", "continuation does not grow the evidence", [("next_token_dist(seq, trans, prior)", "next_token_dist(list(prefix), trans, prior)")]),
    ("4-two-stage-decode", "<EOS> kept in the output", [(
        "        if tok == eos:\n            break\n        seq.append(tok)\n        first.append(tok)",
        "        seq.append(tok)\n        first.append(tok)\n        if tok == eos:\n            break")]),
    ("4-two-stage-decode", "NAR sees its own and later codebooks", [("codes[:, : j - 1].copy()", "codes.copy()")]),
    ("4-two-stage-decode", "off-by-one length cap", [("while len(first) < max_len", "while len(first) <= max_len")]),
    ("4-two-stage-decode", "AR does not see the prompt", [("seq = [int(c) for c in prompt_codes[:, 0]]", "seq = []")]),
    ("4-two-stage-decode", "last NAR stage skipped", [("range(2, K + 1)", "range(2, K)")]),
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
