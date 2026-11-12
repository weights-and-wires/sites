"""The numbers this page quotes from the paper, with the arithmetic that links them.

Every value in the dicts below is typed from the paper (abstract, Section 3,
Section 5 Table 1, Appendix F.2 Table 4, Section 2.4). The script checks the
arithmetic between them.

Run: python3 scripts/paper_numbers.py
"""

best_of_1860 = {"PRM": 78.2, "ORM": 72.4, "majority voting": 69.6}  # Figure 3
print("Best-of-1860 on the 500-problem MATH subset (Figure 3):", best_of_1860)
print(f"PRM minus ORM: {best_of_1860['PRM'] - best_of_1860['ORM']:.1f} points")
print(f"PRM minus majority voting: {best_of_1860['PRM'] - best_of_1860['majority voting']:.1f} points")
print(f"Problems solved at 78.2% of 500: {0.782 * 500:.0f}")

print()
print("PRM800K (Section 2.4): 800K labels, 75K solutions, 12K problems")
print(f"labels per solution: {800_000 / 75_000:.1f}")
print(f"solutions per problem: {75_000 / 12_000:.2f}")
print(f"training problems: 7,500 MATH train + 4,500 MATH test = {7_500 + 4_500}")
print(f"test problems left over: 5,000 - 4,500 = {5_000 - 4_500}")

print()
print("Table 1 (Section 5), best-of-100: ORM, PRM, majority, problems")
table1 = {
    "AP Calculus": (68.9, 86.7, 80.0, 45),
    "AP Chemistry": (68.9, 80.0, 71.7, 60),
    "AP Physics": (77.8, 86.7, 82.2, 45),
    "AMC10/12": (49.1, 53.2, 32.8, 84),
}
total = sum(r[3] for r in table1.values())
print(f"problems in the rows: {total} (Table 1 aggregate row: 234; Section 5 text: 224)")
for i, name in enumerate(("ORM", "PRM", "majority")):
    agg = sum(r[i] * r[3] for r in table1.values()) / total
    print(f"{name} weighted average over rows: {agg:.1f}")
print("Table 1 aggregate row: ORM 63.8, PRM 72.9, majority 61.3")

print()
print("Table 4 (Appendix F.2), best-of-1860 PRM scoring")
table4 = {("product", "neutral=positive"): 78.2, ("minimum", "neutral=positive"): 77.6,
          ("product", "neutral=negative"): 77.4, ("minimum", "neutral=negative"): 77.8}
print(f"spread across the four: {max(table4.values()) - min(table4.values()):.1f} points")
print(f"spread in problems of 500: {(max(table4.values()) - min(table4.values())) * 5:.0f}")
