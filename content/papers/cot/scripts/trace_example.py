"""Prints the numbers for "Exemplars are input, chain of thought, output triples".

Uses the Figure 1 cafeteria question of the paper. The token count is a toy
rule (one token per whitespace-separated word), not a real tokenizer.

Run: python3 content/papers/cot/scripts/trace_example.py
"""

import re

STANDARD = "The answer is 27."  # Figure 1, left: the model output under standard prompting
COT = (
    "The cafeteria had 23 apples originally. They used 20 to make lunch. "
    "So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. "
    "The answer is 9."
)  # Figure 1, right: the model output under chain-of-thought prompting
GOLD = 23 - 20 + 6

EQUATION = re.compile(r"(?<![\d.])(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=\s*(-?\d+)")
ANSWER = re.compile(r"[Tt]he answer is\s*(-?\d+)")


def check(text):
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
    rows = []
    for m in EQUATION.finditer(text):
        a, op, b, c = int(m[1]), m[2], int(m[3]), int(m[4])
        rows.append((m[0], ops[op](a, b), c, ops[op](a, b) == c))
    found = ANSWER.findall(text)
    return rows, int(found[-1]) if found else None


print(f"gold answer: 23 - 20 + 6 = {GOLD}")
for name, text in [("standard", STANDARD), ("chain of thought", COT)]:
    rows, answer = check(text)
    print(f"\n{name}: {len(text.split())} toy tokens, answer {answer}, correct {answer == GOLD}")
    for eq, value, written, ok in rows:
        print(f"  step {eq!r}: computed {value}, written {written}, ok {ok}")
print(f"\nextra generated tokens for the chain: {len(COT.split()) - len(STANDARD.split())}")
