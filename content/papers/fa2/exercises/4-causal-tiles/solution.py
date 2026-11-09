def causal_tile_counts(N, Br, Bc):
    """Classify the tiles of an N x N causal attention matrix.

    Returns (total, skipped, masked, full):
      skipped: every column index is greater than every row index in the tile
      masked:  computed, and some entries need the mask (col > row)
      full:    computed, and no entry needs the mask
    """
    total = skipped = masked = full = 0
    for i in range(0, N, Br):
        row_min, row_max = i, min(i + Br, N) - 1
        for j in range(0, N, Bc):
            col_min, col_max = j, min(j + Bc, N) - 1
            total += 1
            if col_min > row_max:
                skipped += 1
            elif col_max > row_min:
                masked += 1
            else:
                full += 1
    return total, skipped, masked, full


def causal_speedup_bound(N, Br, Bc):
    """Upper bound on the speedup from skipping tiles: total / computed."""
    total, skipped, masked, full = causal_tile_counts(N, Br, Bc)
    return total / (masked + full)
