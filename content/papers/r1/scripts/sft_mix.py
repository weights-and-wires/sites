"""Table 5 sums, and the arithmetic of rejection sampling (Appendix B.3.3)."""

table5 = {
    "Math": (395285, 6094.2),
    "Code": (211129, 7435.7),
    "STEM": (10124, 4928.8),
    "Logic": (10395, 2739.0),
    "General": (177812, 1419.8),
}
total = sum(n for n, _ in table5.values())
reasoning = sum(n for k, (n, _) in table5.items() if k != "General")
print("total samples:", total)
print("reasoning samples (Math + Code + STEM + Logic):", reasoning)
print("non-reasoning samples (General):", table5["General"][0])
print("general share:", round(table5["General"][0] / total, 4))
tokens = sum(n * t for n, t in table5.values()) / total
print("weighted mean tokens per sample:", round(tokens, 1))

print()
# Keep a response only if it is correct and readable. With success rate p per
# sample, the chance that k samples give at least one keeper:
for p in (0.05, 0.3, 0.6):
    for k in (4, 16):
        print(f"p = {p}, k = {k:2d}: P(at least one kept) = {1 - (1 - p) ** k:.4f}")
