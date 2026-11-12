# Our toy: a bigram "language model" over 35 words, a mask on words with an
# 'e', a rule-based "judge", and a GRPO loop. These numbers are ours, not the
# post's.
import numpy as np

rng = np.random.default_rng(0)
V = ["<bos>", "<eos>", "a", "big", "dog", "cat", "can", "run", "jump", "walk",
     "fast", "and", "play", "with", "you", "sits", "on", "mat", "to", "go",
     "home", "today", "in", "my", "yard", "the", "here", "see", "eat", "green",
     "fresh", "where", "there", "keep", "we"]
n, idx = len(V), {w: i for i, w in enumerate(V)}
has_e = np.array([("e" in w) and not w.startswith("<") for w in V])
NOUN = ["dog", "cat", "mat", "yard"]
ok = {  # which next words read as normal English
    "<bos>": ["a", "my", "the", "we", "you", "here"],
    "a": NOUN + ["big", "green", "fresh"], "the": NOUN + ["big", "green", "fresh"],
    "my": NOUN + ["green"], "big": NOUN, "green": NOUN, "fresh": ["mat", "yard"],
    "dog": ["can", "sits", "run", "and", "<eos>", "keep"],
    "cat": ["can", "sits", "run", "and", "<eos>", "keep"],
    "can": ["run", "jump", "walk", "go", "play", "see", "eat"],
    "we": ["can", "go", "play", "see", "eat", "keep"],
    "you": ["can", "go", "play", "see", "eat"],
    "here": ["we", "you", "a", "the"], "there": ["we", "you", "a", "the"],
    "where": ["we", "you", "a", "the"],
    "run": ["fast", "and", "to", "<eos>", "today", "there"],
    "jump": ["fast", "and", "<eos>", "on", "here"],
    "walk": ["to", "fast", "<eos>", "and", "here"],
    "go": ["to", "home", "<eos>", "there", "here"],
    "to": ["home", "go", "a", "my", "the"], "sits": ["on", "in", "here", "there"],
    "on": ["a", "my", "the"], "in": ["a", "my", "the"],
    "home": ["<eos>", "and", "today"], "yard": ["<eos>", "and", "today", "here"],
    "mat": ["<eos>", "and", "today"], "fast": ["<eos>", "and", "today"],
    "today": ["<eos>"], "and": ["play", "go", "run", "jump", "walk", "see", "eat"],
    "play": ["with", "fast", "<eos>", "here"], "with": ["you", "a", "my", "the"],
    "see": ["a", "the", "my", "you"], "eat": ["a", "the", "my", "fresh", "green"],
    "keep": ["a", "my", "the"],
}
GOOD = np.zeros((n, n), bool)
for a, bs in ok.items():
    for b in bs:
        GOOD[idx[a], idx[b]] = True
EOS = idx["<eos>"]


def probs(logits, u, mask):
    l = logits[u].copy()
    if mask:
        l[has_e] = -np.inf          # the logit mask
    l[idx["<bos>"]] = -np.inf
    p = np.exp(l - l.max())
    return p / p.sum()


def sample(logits, mask, maxlen=8):
    seq = [idx["<bos>"]]
    for _ in range(maxlen):
        w = rng.choice(n, p=probs(logits, seq[-1], mask))
        seq.append(w)
        if w == EOS:
            break
    return seq


def judge(seq):
    pairs = list(zip(seq[:-1], seq[1:]))
    coherence = 1 + round(4 * np.mean([GOOD[a, b] for a, b in pairs]))
    responds = 1 + min(4, len(seq) - 2)   # three or more words score 4
    return coherence, responds


def usable(seq):
    c, r = judge(seq)
    return (not any(has_e[w] for w in seq)) and c >= 4 and r >= 4


def pretrained():
    return np.where(GOOD, 2.0, 0.0) + rng.normal(0, 0.3, (n, n))


def measure(logits, mask, N=200):
    qs = [sample(logits, mask) for _ in range(N)]
    return float(np.mean([usable(q) for q in qs])), float(np.mean([len(q) - 2 for q in qs]))


def grpo(logits, mask=True, steps=150, G=8, P=4, lr=0.5, reward="sum", every=15):
    logits = logits.copy()
    trace = [measure(logits, mask)]
    for step in range(1, steps + 1):
        for _ in range(P):
            seqs = [sample(logits, mask) for _ in range(G)]
            cr = np.array([judge(q) for q in seqs])
            if reward == "sum":
                rew = (cr[:, 0] - 1 + cr[:, 1] - 1) / 8
            else:
                rew = (cr.min(1) - 1) / 4
            adv = (rew - rew.mean()) / (rew.std() + 1e-6)
            for q, a in zip(seqs, adv):
                for u, v in zip(q[:-1], q[1:]):
                    g = -probs(logits, u, mask)
                    g[v] += 1
                    if mask:
                        g[has_e] = 0
                    logits[u] += lr * a * g / (G * P)
        if step % every == 0:
            trace.append(measure(logits, mask))
    return logits, trace


base = pretrained()
print(f"unmasked, instructed only : {measure(base, False)[0]:.0%} usable")
print(f"masked, no training       : {measure(base, True)[0]:.0%} usable")
tuned, tr_sum = grpo(base, reward="sum")
print(f"masked + GRPO (sum)       : {tr_sum[-1][0]:.0%} usable, "
      f"{tr_sum[0][1]:.1f} -> {tr_sum[-1][1]:.1f} words per answer")
print("sample:", " ".join(V[w] for w in sample(tuned, True)[1:]))
_, tr_min = grpo(base, reward="min")
print(f"masked + GRPO (min)       : {tr_min[-1][0]:.0%} usable, "
      f"{tr_min[0][1]:.1f} -> {tr_min[-1][1]:.1f} words per answer")

result = {"every": 15, "sum": tr_sum, "min": tr_min}
