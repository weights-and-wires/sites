"""Prints the numbers for "Emergence with model scale".

The accuracies are copied from Table 2 of the paper (GSM8K and SVAMP, %).
The script computes the chain-of-thought gain per model size and the
emergence point used in exercise 3: the first size from which every larger
size gains.

Run: python3 content/papers/cot/scripts/scale_gains.py
"""

TABLE_2 = {
    # name: (sizes in billions, standard, chain of thought)
    "GSM8K": {
        "LaMDA": ([0.42, 2, 8, 68, 137], [2.6, 3.6, 3.2, 5.7, 6.5], [0.4, 1.9, 1.6, 8.2, 14.3]),
        "GPT": ([0.35, 1.3, 6.7, 175], [2.2, 2.4, 4.0, 15.6], [0.5, 0.5, 2.4, 46.9]),
        "PaLM": ([8, 62, 540], [4.9, 9.6, 17.9], [4.1, 29.9, 56.9]),
    },
    "SVAMP": {
        "PaLM": ([8, 62, 540], [15.1, 48.2, 69.4], [16.8, 46.7, 79.0]),
    },
}


def emergence_index(gains):
    start = None
    for i, g in enumerate(gains):
        if g > 0:
            start = i if start is None else start
        else:
            start = None
    return start


for dataset, models in TABLE_2.items():
    print(f"{dataset}")
    for name, (sizes, std, cot) in models.items():
        gains = [round(c - s, 1) for s, c in zip(std, cot)]
        i = emergence_index(gains)
        first_positive = next((j for j, g in enumerate(gains) if g > 0), None)
        print(f"  {name}: sizes {sizes}")
        print(f"    gains {gains}")
        print(f"    emergence size {sizes[i] if i is not None else None}B; first positive gain at {sizes[first_positive]}B")

sizes, std, cot = TABLE_2["GSM8K"]["GPT"]
print(f"\nGPT 175B on GSM8K: {cot[-1]} / {std[-1]} = {cot[-1] / std[-1]:.2f} times the standard accuracy")
sizes, std, cot = TABLE_2["GSM8K"]["PaLM"]
print(f"PaLM 540B on GSM8K: {cot[-1]} / {std[-1]} = {cot[-1] / std[-1]:.2f} times the standard accuracy")
