"""Recompute the relative changes that the page quotes from the paper's tables.

Every raw number below is copied from a table in the paper (arXiv 1512.02595v1).
The script derives the percentages.

Run: python3 scripts/paper_tables.py
"""


def rel(old, new):
    return 100 * (old - new) / old


# Table 1, dev WER of the deepest model (7 RNN, 9 total), without and with BatchNorm.
print(f"Table 1, 9-layer: BatchNorm {rel(10.83, 9.52):.1f}% relative (10.83 to 9.52)")
print(f"Table 1, 5-layer 1 RNN: BatchNorm {rel(13.55, 14.40):.1f}% relative (13.55 to 14.40; negative means worse)")

# Table 2, dev WER, 9 layers with 7 recurrent layers.
print(f"Table 2: SortaGrad without BatchNorm {rel(11.96, 10.83):.1f}% (11.96 to 10.83), with BatchNorm {rel(9.78, 9.52):.1f}% (9.78 to 9.52)")

# Table 4, noisy dev, 1-layer 1D against 3-layer 2D.
print(f"Table 4: 1-layer 1D to 3-layer 2D on noisy dev {rel(19.36, 14.74):.1f}% (19.36 to 14.74)")

# Table 5, 4x stride, dev with LM.
print(f"Table 5: stride 4 with LM, unigrams 11.92 and bigrams 9.93: bigrams are {rel(11.92, 9.93):.1f}% better")
print(f"Table 5: stride 2 with LM, unigrams 9.52 and bigrams 9.66: bigrams are {rel(9.52, 9.66):.1f}% (negative means worse)")

# Table 6, language model gain.
print(f"Table 6, English 5-layer 1 RNN: LM gain {rel(27.79, 14.39):.1f}% (27.79 to 14.39)")
print(f"Table 6, English 9-layer 7 RNN: LM gain {rel(14.93, 9.52):.1f}% (14.93 to 9.52)")
print(f"Table 6, Mandarin 5-layer 1 RNN: LM gain {rel(9.80, 7.13):.1f}% (9.80 to 7.13)")
print(f"Table 6, Mandarin 9-layer 7 RNN: LM gain {rel(7.55, 5.81):.1f}% (7.55 to 5.81)")

# Table 12, DS1 to DS2.
print(f"Table 12: DS1 to DS2 on the Baidu test set {rel(24.01, 13.59):.1f}% (24.01 to 13.59)")

# Table 16, Mandarin test set, shallow to deepest.
print(f"Table 16: 5-layer 1 RNN to 9-layer 7 RNN + BatchNorm + 2D conv on the test set {rel(15.41, 7.93):.1f}% (15.41 to 7.93)")

# Section 7.4, deployed against research system.
print(f"Section 7.4: deployed CER 6.10 against research 5.81: {100 * (6.10 - 5.81) / 5.81:.1f}% relative degradation")

# Section 4.2, Table 8: CPU to GPU CTC.
print(f"Table 8: English CTC time {5888.12 / 203.56:.1f}x faster on GPU (5888.12 s to 203.56 s)")

# Table 13, read speech, DS2 against human.
for name, ds2, human in (("WSJ eval'92", 3.60, 5.03), ("WSJ eval'93", 4.98, 8.08), ("LibriSpeech test-clean", 5.33, 5.83), ("LibriSpeech test-other", 13.25, 12.69)):
    verdict = "better than" if ds2 < human else "worse than"
    print(f"Table 13, {name}: DS2 {ds2} is {verdict} human {human}")

# Constants quoted from the text of the paper (Sections 1, 3.6, 5.3, 7.4 and the abstract).
print()
hours_en, hours_zh = 11940, 9400
print(f"Training data: English {hours_en:,} h, Mandarin {hours_zh:,} h, total {hours_en + hours_zh:,} h")
print(f"Characters per second: English 14.1, Mandarin 3.3 (ratio {14.1 / 3.3:.1f})")
print(f"Power law (Section 5.3): about 40% relative WER drop per 10x data, so each 10x multiplies WER by 0.6; 100x gives {0.6 ** 2:.2f}")
print("Training scale (Section 1 and abstract): 7x speedup over the previous system; 3 to 5 days per model on 8 or 16 GPUs")
print(f"Deployed model (Section 7.4): five forward-only layers of 2560 units, row convolution with tau = 19, CER {6.10} vs {5.81}")
