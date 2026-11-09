#!/usr/bin/env python3
"""Check that the exercise tests catch plausible wrong implementations.

For each mutant, this script copies an exercise's solution, applies one
realistic bug by string replacement, runs the exercise's tests through
static/runner/harness.py, and prints the tests that fail. A mutant that fails
no test is a gap in the tests, and the script exits with status 1.

Run from the repository root:

    python3 content/papers/r1/scripts/check_mutants.py
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
# Not a mutant: dropping the -pi term in exercise 3. With group-mean advantages that sum to 0,
# the -pi terms cancel, so the step is identical.
MUTANTS = [
    ("1-rule-based-rewards", "first answer tag instead of the last", [("return found[-1].strip() if found else None", "return found[0].strip() if found else None")]),
    ("1-rule-based-rewards", "substring match instead of integer equality", [("return 1.0 if int(ans) == gold else 0.0", "return 1.0 if str(gold) in ans else 0.0")]),
    ("1-rule-based-rewards", "answer found anywhere in the text, not in the tag", [("ans = extract_answer(text)\n    if ans is None", "nums = re.findall(r\"-?\\d+\", text)\n    ans = nums[-1] if nums else None\n    if ans is None")]),
    ("1-rule-based-rewards", "accuracy requires the format too", [("return accuracy_reward(text, gold) + format_reward(text)", "return accuracy_reward(text, gold) * (1 + format_reward(text))")]),
    ("1-rule-based-rewards", "accuracy weighted twice the format", [("return accuracy_reward(text, gold) + format_reward(text)", "return 2 * accuracy_reward(text, gold) + format_reward(text)")]),
    ("1-rule-based-rewards", "format accepts trailing text", [("m = THINK_ANSWER.fullmatch(body)", "m = THINK_ANSWER.match(body)")]),
    ("1-rule-based-rewards", "format accepts a blank think block", [("if m is None or not m.group(1).strip() or not m.group(2).strip():", "if m is None:")]),
    ("1-rule-based-rewards", "format accepts repeated tags", [("        if body.count(tag) != 1:\n            return 0.0\n", "        if body.count(tag) < 1:\n            return 0.0\n")]),
    ("1-rule-based-rewards", "no DOTALL, so multi-line blocks fail", [("ANSWER_TAG = re.compile(r\"<answer>(.*?)</answer>\", re.DOTALL)", "ANSWER_TAG = re.compile(r\"<answer>(.*?)</answer>\")"), ("THINK_ANSWER = re.compile(r\"<think>(.+?)</think>\\s*<answer>(.+?)</answer>\", re.DOTALL)", "THINK_ANSWER = re.compile(r\"<think>(.+?)</think>\\s*<answer>(.+?)</answer>\")")]),
    ("2-language-and-rejection", "numbers and symbols counted as non-target words", [("words = [w for w in cot.split() if any(c.isalpha() for c in w)]", "words = cot.split()")]),
    ("2-language-and-rejection", "characters instead of words", [("    return len(target) / len(words)", "    return sum(len(w) for w in target) / sum(len(w) for w in words)")]),
    ("2-language-and-rejection", "a word is target if any letter is ASCII", [("all(c.isascii() for c in w if c.isalpha())", "any(c.isascii() for c in w if c.isalpha())")]),
    ("2-language-and-rejection", "mixed language allowed above 0.5", [("if language_reward(think) != 1.0:", "if language_reward(think) < 0.5:")]),
    ("2-language-and-rejection", "code fences not filtered", [("    if \"```\" in think:\n        return False\n", "")]),
    ("2-language-and-rejection", "long paragraphs measured over the whole thinking", [("any(len(p.split()) > max_paragraph_words for p in think.split(\"\\n\\n\"))", "len(think.split()) > max_paragraph_words")]),
    ("2-language-and-rejection", "paragraph limit off by one", [("len(p.split()) > max_paragraph_words", "len(p.split()) >= max_paragraph_words")]),
    ("2-language-and-rejection", "wrong answers are kept", [("if not re.fullmatch(r\"-?\\d+\", answer) or int(answer) != gold:", "if not re.fullmatch(r\"-?\\d+\", answer):")]),
    ("2-language-and-rejection", "rejection sampling ignores k", [("        if len(kept) == k:\n            break\n", "")]),
    ("2-language-and-rejection", "rejection sampling reorders (longest first)", [("    return kept", "    return sorted(kept, key=len, reverse=True)")]),
    ("3-toy-grpo", "gradient ascent turned into descent", [("return logits + lr * grad / len(r)", "return logits - lr * grad / len(r)")]),
    ("3-toy-grpo", "no group baseline or scaling (raw rewards)", [("adv = (r - r.mean()) / (r.std() + 1e-6)", "adv = r")]),
    ("3-toy-grpo", "no std normalization", [("adv = (r - r.mean()) / (r.std() + 1e-6)", "adv = r - r.mean()")]),
    ("3-toy-grpo", "sample std (ddof=1)", [("r.std() + 1e-6", "r.std(ddof=1) + 1e-6")]),
    ("3-toy-grpo", "gradient forgets the one-hot term", [("        g = -p.copy()\n        g[k] += 1.0", "        g = -p.copy()")]),
    ("3-toy-grpo", "sum over the group instead of the mean", [("return logits + lr * grad / len(r)", "return logits + lr * grad")]),
    ("3-toy-grpo", "update modifies the input in place", [("return logits + lr * grad / len(r)", "logits += lr * grad / len(r)\n    return logits")]),
    ("3-toy-grpo", "no guard, so equal rewards give nan", [("r.std() + 1e-6", "r.std()")]),
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
