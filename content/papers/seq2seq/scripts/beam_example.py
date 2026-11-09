"""Greedy decoding versus beam search (Section 3.2) on a table model."""

import numpy as np

A, B, EOS = 0, 1, 2
NAMES = ["a", "b", "<EOS>"]

TABLE = {
    (): [0.6, 0.4, 0.0],
    (A,): [0.30, 0.25, 0.45],
    (B,): [0.05, 0.05, 0.90],
    (A, A): [0.35, 0.25, 0.40],
}
DEFAULT = [0.3, 0.2, 0.5]


def step_logprobs(prefix):
    p = np.array(TABLE.get(tuple(prefix), DEFAULT), dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(p)


def beam_search(step_fn, beam_size, eos, max_len):
    beam, done = [((), 0.0)], []
    for _ in range(max_len):
        cands = []
        for prefix, score in beam:
            lp = step_fn(prefix)
            for tok in range(len(lp)):
                cands.append((prefix + (tok,), score + lp[tok]))
        cands.sort(key=lambda c: -c[1])
        beam = []
        for prefix, score in cands[:beam_size]:
            (done if prefix[-1] == eos else beam).append((prefix, score))
        if not beam:
            break
    return sorted(done, key=lambda c: -c[1])


def show(seq):
    return " ".join(NAMES[t] for t in seq)


if __name__ == "__main__":
    for bs in (1, 2):
        res = beam_search(step_logprobs, bs, EOS, max_len=6)
        print(f"beam size {bs}: {len(res)} complete hypotheses")
        for seq, s in res[:3]:
            print(f"   {show(seq):22s} log p = {s:.4f}  p = {np.exp(s):.4f}")
