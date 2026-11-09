"""Numbers for the language-modeling example (Section 2, Equation 1).

Run: python3 scripts/lm_factorization.py
"""

import math

# Conditional probabilities a toy model assigns to the sequence "the cat sat".
cond = [("p(the)", 0.5), ("p(cat | the)", 0.4), ("p(sat | the cat)", 0.5)]
p = 1.0
for name, v in cond:
    p *= v
    print(f"{name:<18} = {v}")
print(f"p(x) = product = {p:.4f}")

nll = [-math.log(v) for _, v in cond]
print("per-token NLL (nats) =", [round(v, 4) for v in nll])
avg = sum(nll) / len(nll)
print(f"average NLL = {avg:.4f} nats")
print(f"perplexity = exp(average NLL) = {math.exp(avg):.4f}")
print(f"check: p(x)^(-1/3) = {p ** (-1 / 3):.4f}")

V = 50257
print(f"\nuniform model over V = {V}: perplexity = {math.exp(-math.log(1 / V)):.0f}")

# The same total log-probability, expressed per byte instead of per token.
n_bytes = len("the cat sat".encode("utf-8"))
bpb = -math.log2(p) / n_bytes
print(f"\n'the cat sat' is {n_bytes} bytes; bits per byte = {bpb:.4f}")

# Section 2: a supervised example is the same sequence with the loss on a suffix.
# Sequence: prompt tokens then answer tokens. Chain rule splits the log-probability.
logp = [-1.2, -0.7, -2.0, -0.3, -0.5]  # tokens 0..4
prompt, answer = logp[:3], logp[3:]
print("\nfull-sequence log p       =", round(sum(logp), 4))
print("log p(prompt)             =", round(sum(prompt), 4))
print("log p(answer | prompt)    =", round(sum(answer), 4))
print("prompt + answer           =", round(sum(prompt) + sum(answer), 4))
