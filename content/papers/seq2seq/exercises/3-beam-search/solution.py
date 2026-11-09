import numpy as np


def beam_search(step_fn, beam_size, eos, max_len):
    """Return complete hypotheses [(tokens, log_prob), ...], best first."""
    beam, done = [((), 0.0)], []
    for _ in range(max_len):
        cands = []
        for prefix, score in beam:
            lp = step_fn(prefix)
            for tok in range(len(lp)):
                cands.append((prefix + (tok,), score + float(lp[tok])))
        cands.sort(key=lambda c: -c[1])
        beam = []
        for prefix, score in cands[:beam_size]:
            if prefix[-1] == eos:
                done.append((prefix, score))
            else:
                beam.append((prefix, score))
        if not beam:
            break
    return sorted(done, key=lambda c: -c[1])
