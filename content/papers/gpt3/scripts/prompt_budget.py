"""Few-shot prompt layout and context budget (Section 2, Section 2.4, Appendix G).

The paper's arithmetic prompts look like "Q: What is 98 plus 45?" then "A:"
(Figure G.44), with demonstrations separated by newlines (Section 2.4). The
paper doesn't say how many tokens a demonstration takes, so this script uses a
toy tokenizer: a run of letters is one token, each digit is one token, each
other non-space character is one token, and a run of newlines is one token.
GPT-3's real byte-level BPE counts differ.

Run: python3 scripts/prompt_budget.py
"""

import re

N_CTX = 2048  # Section 2.1: every model has a 2048-token context window
TOKEN = re.compile(r"[A-Za-z]+|\d|\n+|[^\sA-Za-z\d]")


def n_tokens(text):
    return len(TOKEN.findall(text))


def demo(q, a):
    return f"Q: {q}\nA: {a}\n\n"


def query(q):
    return f"Q: {q}\nA:"


demos = [("What is 48 plus 76?", "124"), ("What is 17 plus 25?", "42"), ("What is 90 plus 9?", "99")]
final = "What is 98 plus 45?"

for k in (0, 1, 3):
    prompt = "".join(demo(q, a) for q, a in demos[:k]) + query(final)
    print(f"--- K = {k}: {n_tokens(prompt)} tokens ---")
    print(prompt)
    print()

per_demo = [n_tokens(demo(q, a)) for q, a in demos]
print("tokens per demonstration:", per_demo)
avg = sum(per_demo) / len(per_demo)
q_tokens = n_tokens(query(final))
answer_tokens = 3  # reserve room for the completion (assumption)
k_max = int((N_CTX - q_tokens - answer_tokens) // avg)
print(f"final query: {q_tokens} tokens; average demonstration: {avg:.2f} tokens")
print(f"largest K that fits in n_ctx = {N_CTX} with {answer_tokens} tokens reserved: {k_max}")

# A longer task takes fewer demonstrations. Suppose a demonstration takes 150 tokens.
for L in (20, 60, 150):
    print(f"demonstration of {L} tokens -> K_max = {(N_CTX - 60) // L}")
