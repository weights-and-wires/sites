"""Table 3 (AIME 2024 pass@1 by stage) and the distillation comparison of Table 16."""

stages = ["R1-Zero", "R1-Dev1", "R1-Dev2", "R1-Dev3", "R1"]
aime = [77.9, 59.0, 74.0, 78.1, 79.8]
alpaca = [24.7, 50.1, 55.8, 62.1, 87.6]
print("AIME 2024 pass@1 and change from the previous stage")
for i, s in enumerate(stages):
    d = "" if i == 0 else f"{aime[i] - aime[i - 1]:+.1f}"
    print(f"  {s:8s} {aime[i]:5.1f} {d}")
print("AlpacaEval 2.0 LC-winrate, R1 vs Dev3:", round(alpaca[4] - alpaca[3], 1), "points,", round((alpaca[4] / alpaca[3] - 1) * 100, 1), "% relative")

print()
rows = {
    "Qwen2.5-32B-Zero (RL only)": (47.0, 91.6, 55.0, 40.2),
    "R1-Distill-Qwen-32B (SFT on 800k)": (72.6, 94.3, 62.1, 57.2),
}
names = ("AIME 2024 pass@1", "MATH-500 pass@1", "GPQA Diamond", "LiveCodeBench")
a, b = rows.values()
for n, x, y in zip(names, a, b):
    print(f"  {n:18s} RL {x:5.1f}  distilled {y:5.1f}  gap {y - x:+.1f}")
