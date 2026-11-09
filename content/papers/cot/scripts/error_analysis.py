"""Prints the numbers for "Chains can be wrong, and they generalize in length".

Part 1 turns the percentages of the manual error analysis (Section 3.2) into
counts of the 50 sampled examples. Part 2 builds the last-letter chains of
Figure 3 for names of two, three, and four words and counts their steps. Part 3
reads the length-generalization numbers from Table 5 (PaLM 540B, %).

Run: python3 content/papers/cot/scripts/error_analysis.py
"""

SAMPLE = 50
print("Section 3.2, LaMDA 137B on GSM8K, 50 sampled examples per group")
print(f"  correct answers: {SAMPLE - 2} with a correct chain, 2 by coincidence")
almost = round(0.46 * SAMPLE)
print(f"  wrong answers: 46% almost correct = {almost} examples, 54% major errors = {SAMPLE - almost} examples")


def last_letter_chain(words):
    """The chain-of-thought template of Figure 3, extended to any number of words."""
    lines = [f'The last letter of "{w}" is "{w[-1]}".' for w in words]
    out = "".join(w[-1] for w in words)
    return " ".join(lines) + f' Concatenating them is "{out}". So the answer is {out}.', out


print("\nLast-letter concatenation chains (the exemplars in Figure 3 have two words)")
for words in (["Lady", "Gaga"], ["Amy", "Brown", "Lee"], ["Amy", "Brown", "Lee", "Ng"]):
    chain, answer = last_letter_chain(words)
    print(f"  {len(words)} words -> answer {answer!r}, chain of {len(chain.split())} words, {len(words)} lookup steps")

print("\nCoin flip: the parity rule of Figure 3 (heads up after an even number of flips)")
for flips in (1, 2, 3, 4):
    print(f"  {flips} flips -> heads up: {flips % 2 == 0}")

TABLE_5 = {  # PaLM 540B: (standard, chain of thought)
    "last letters, 2 words (in domain)": (7.6, 99.4),
    "last letters, 3 words (OOD)": (0.2, 94.8),
    "last letters, 4 words (OOD)": (0.0, 63.0),
    "coin flip, 2 flips (in domain)": (98.1, 100.0),
    "coin flip, 3 flips (OOD)": (49.3, 98.6),
    "coin flip, 4 flips (OOD)": (54.8, 90.2),
}
print("\nTable 5, PaLM 540B")
for name, (s, c) in TABLE_5.items():
    print(f"  {name:<36} standard {s:5.1f}  chain of thought {c:5.1f}  gain {c - s:+6.1f}")
