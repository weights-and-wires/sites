#!/usr/bin/env python3
"""How many times H^-1 must be updated with Equation 3 (Section 4, Step 1)."""

for d_row, d_col in ((16, 32), (4096, 4096)):
    obq = d_row * d_col  # one update per weight, each row has its own order
    gptq = d_col  # one update per column, shared by all rows
    print(f"d_row={d_row}, d_col={d_col}: OBQ {obq:,} updates, GPTQ {gptq:,} updates, ratio {obq // gptq:,} = min(d_row, d_col)? {obq // gptq == min(d_row, d_col)}")
