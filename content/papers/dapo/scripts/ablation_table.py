"""Print the gain from each technique in Table 1 (Section 4.2), AIME 2024 avg@32.

Run: python3 scripts/ablation_table.py
"""

rows = [
    ("Naive GRPO", 30),
    ("+ Overlong filtering", 36),
    ("+ Clip-Higher", 38),
    ("+ Soft overlong punishment", 41),
    ("+ Token-level loss", 42),
    ("+ Dynamic sampling (DAPO)", 50),
]
prev = None
for name, score in rows:
    gain = "" if prev is None else f"  ({score - prev:+d})"
    print(f"{name:<30} {score}{gain}")
    prev = score
print(f"Total gain over naive GRPO: {rows[-1][1] - rows[0][1]:+d}")
print(f"DeepSeek-R1-Zero-Qwen-32B: 47, DAPO margin: {rows[-1][1] - 47:+d}")
