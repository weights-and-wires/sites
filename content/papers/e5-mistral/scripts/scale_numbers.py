#!/usr/bin/env python3
"""Arithmetic on the training and storage numbers in Sections 3.2, 4.2, and Appendix A.

Run from the paper folder: python3 scripts/scale_numbers.py
"""

examples = 1.8e6   # about 1.8M training examples after sampling (Section 4.2)
batch = 2048       # batch size (Appendix A)
print(f"steps in one epoch: {examples / batch:.0f}")

trainable = 42e6   # LoRA parameters (Appendix A)
base = 7e9         # about 7B parameters in the Mistral-7B base (rounded)
print(f"trainable share: {100 * trainable / base:.1f} percent")

# If every document and hard negative in the batch is a candidate (an implementation
# choice that the exercise makes, not a statement in the paper):
print(f"candidates per query: {batch + batch} (positive, {batch - 1} in-batch, {batch} hard)")

dims = 4096        # embedding width (Limitations)
per_doc = dims * 4
print(f"float32 bytes per document: {per_doc}")
print(f"gigabytes for 1M documents: {per_doc * 1e6 / 1e9:.1f}")

print(f"MTEB gain over the 64.2 baseline: {66.6 - 64.2:.1f}")
print(f"instruction ablation: {64.5 - 60.3:.1f} points")
