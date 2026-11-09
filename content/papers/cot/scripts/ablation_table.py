"""Prints the numbers for the ablation table in "Intermediate tokens as extra computation".

The values are copied from Table 6 of the paper (LaMDA 137B, accuracy in %).
The script computes each variant's difference from standard prompting.

Run: python3 content/papers/cot/scripts/ablation_table.py
"""

DATASETS = ["GSM8K", "SVAMP", "ASDiv", "MAWPS"]
TABLE_6 = {
    "standard prompting": [6.5, 29.5, 40.1, 43.2],
    "equation only": [5.4, 35.1, 45.9, 50.1],
    "variable compute only": [6.4, 28.0, 39.4, 41.3],
    "reasoning after answer": [6.1, 30.7, 38.6, 43.6],
    "chain of thought": [14.3, 36.7, 46.6, 57.9],
}
TABLE_2_COT = [14.3, 37.5, 46.6, 57.9]  # Table 2, LaMDA 137B, chain of thought

base = TABLE_6["standard prompting"]
print(f"{'variant':<24}" + "".join(f"{d:>14}" for d in DATASETS))
for name, row in TABLE_6.items():
    cells = [f"{v:5.1f} ({v - b:+5.1f})" if name != "standard prompting" else f"{v:5.1f}        " for v, b in zip(row, base)]
    print(f"{name:<24}" + "".join(f"{c:>14}" for c in cells))

print("\nmean gain over standard, across the four datasets")
for name, row in TABLE_6.items():
    if name == "standard prompting":
        continue
    print(f"  {name:<24}{sum(v - b for v, b in zip(row, base)) / 4:+6.2f}")

print("\nTable 6 against Table 2 for chain of thought:")
for d, a, b in zip(DATASETS, TABLE_6["chain of thought"], TABLE_2_COT):
    note = "" if a == b else "  <- differs"
    print(f"  {d:<6} Table 6 {a:5.1f}   Table 2 {b:5.1f}{note}")
