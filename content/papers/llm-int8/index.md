---
id: "llm-int8"
title: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
authors: "Dettmers et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2208.07339"
url: "https://arxiv.org/abs/2208.07339"
track: inference
tier: extended
est_hours: 3
status: stub
summary: "Reveals emergent outlier features and the mixed-precision decomposition that handles them."
related_course: "inference: quantization"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
  - id: mixed-precision
    reason: "FP32 master copy of the weights"
background: [quantization, outliers]
course_links:
  - title: "Quantization on Ampere"
    url: "https://qwen.ifkash.dev/c/18-quantization"
  - title: "Quantized kernels"
    url: "https://gpu.ifkash.dev/c/18-quantized-kernels"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Vector-wise absmax quantization
- Outlier dimensions kept in FP16
- Quality holds up to 175B
