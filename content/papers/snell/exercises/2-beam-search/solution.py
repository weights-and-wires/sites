def beam_search(expand, score, n_beams, beam_width, n_steps):
    """PRM beam search with `n_beams` beams and beam width `beam_width`.

    Returns the final `n_beams` prefixes sorted by score, best first.
    """
    beams = expand((), n_beams)
    for _ in range(n_steps - 1):
        ranked = sorted(beams, key=lambda p: -score(p))
        kept = ranked[: n_beams // beam_width]
        beams = []
        for prefix in kept:
            beams.extend(expand(prefix, beam_width))
    return sorted(beams, key=lambda p: -score(p))
