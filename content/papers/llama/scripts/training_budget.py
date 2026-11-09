"""Compute the training-budget numbers for the LLaMA page.

Run: python3 scripts/training_budget.py

Inputs from the paper: Table 2 (parameters, tokens), Section 2.4 (380
tokens/sec/GPU, 2048 A100 GPUs, 1.4T tokens, about 21 days), Section 6 and
Table 15 (400 W, PUE 1.1, 0.385 kg CO2eq/KWh, 1,022,362 GPU-hours for 65B).
"""

MODELS = [("7B", 6.7e9, 1.0e12), ("13B", 13.0e9, 1.0e12), ("33B", 32.5e9, 1.4e12), ("65B", 65.2e9, 1.4e12)]

print("tokens per parameter, and the Chinchilla-style 20 tokens per parameter:")
for name, n, d in MODELS:
    print(f"  {name}: {d / n:6.1f} tokens/param; 20 N = {20 * n / 1e9:6.0f}B tokens; trained on {d / 20 / n:4.1f}x that")

print()
print("training FLOPs, C = 6 N D:")
for name, n, d in MODELS:
    print(f"  {name}: {6 * n * d:.2e}")

print()
tokens, rate, gpus = 1.4e12, 380, 2048
seconds = tokens / (rate * gpus)
print(f"65B throughput: {rate * gpus:,} tokens/s; 1.4T tokens take {seconds / 86400:.1f} days")
gpu_hours = gpus * seconds / 3600
print(f"GPU-hours at that pace: {gpu_hours:,.0f} (Table 15: 1,022,362)")
mwh = 1_022_362 * 400 * 1.1 / 1e6
print(f"Table 15 energy: 1,022,362 h x 400 W x 1.1 = {mwh:.1f} MWh; x 0.385 = {mwh * 0.385:.1f} tCO2eq")
print(f"model FLOPs per GPU: 6 N x 380 = {6 * 65.2e9 * rate / 1e12:.1f} TFLOP/s")
print(f"  as a share of the A100 bf16 peak of 312 TFLOP/s: {6 * 65.2e9 * rate / 312e12:.0%} (the peak is not from the paper)")

print()
print("optimizer steps at a 4M-token batch:", f"{1.4e12 / 4e6:,.0f} for 1.4T tokens; {1.0e12 / 4e6:,.0f} for 1.0T tokens")

print()
print("Lifetime compute, hypothetical: 10B on 200B tokens versus 7B on 1T tokens, assuming equal quality.")
a_n, a_d, b_n, b_d = 10e9, 200e9, 7e9, 1e12
ta, tb = 6 * a_n * a_d, 6 * b_n * b_d
print(f"  train: {ta:.2e} vs {tb:.2e}; extra for the smaller model: {tb - ta:.2e}")
print(f"  serve: 2 N FLOPs per token: {2 * a_n:.1e} vs {2 * b_n:.1e}; saving per served token {2 * (a_n - b_n):.1e}")
breakeven = (tb - ta) / (2 * (a_n - b_n))
print(f"  break-even: {breakeven:.2e} served tokens")
for served in (1e12, 5e12, 2e13):
    print(f"  serve {served:.0e} tokens: total {ta + 2 * a_n * served:.2e} vs {tb + 2 * b_n * served:.2e}")

print()
print("Weight memory at 2 bytes per parameter (fp16 or bf16), weights only:")
for name, n in [("GPT-3 175B", 175e9), ("LLaMA-65B", 65.2e9), ("LLaMA-13B", 13.0e9), ("LLaMA-7B", 6.7e9)]:
    gb = 2 * n / 1e9
    print(f"  {name}: {gb:.1f} GB; 80 GB GPUs needed at least: {-(-gb // 80):.0f}; fits one 32 GB GPU: {gb < 32}")
