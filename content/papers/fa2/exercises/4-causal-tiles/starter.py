def causal_tile_counts(N, Br, Bc):
    """Classify the tiles of an N x N causal attention matrix.

    Returns (total, skipped, masked, full):
      skipped: every column index is greater than every row index in the tile
      masked:  computed, and some entries need the mask (col > row)
      full:    computed, and no entry needs the mask
    """
    raise NotImplementedError


def causal_speedup_bound(N, Br, Bc):
    """Upper bound on the speedup from skipping tiles: total / computed."""
    raise NotImplementedError
