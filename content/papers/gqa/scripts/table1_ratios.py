"""Compute the ratios quoted from Table 1 of the paper.

The values are copied from Table 1 (Tinfer, then the Average column).

Run: python3 scripts/table1_ratios.py
"""

rows = {
    "MHA-Large": (0.37, 46.0),
    "MHA-XXL": (1.51, 47.2),
    "MQA-XXL": (0.24, 46.6),
    "GQA-8-XXL": (0.28, 47.1),
}
base_t, base_q = rows["MHA-XXL"]
for name, (t, q) in rows.items():
    print(f"{name}: Tinfer {t}, average {q}, speedup over MHA-XXL {base_t / t:.2f}x, quality gap {q - base_q:+.1f}")
print(f"GQA-8-XXL time over MQA-XXL: {rows['GQA-8-XXL'][0] / rows['MQA-XXL'][0]:.2f}x")
