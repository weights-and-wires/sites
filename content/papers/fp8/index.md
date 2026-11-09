---
id: "fp8"
title: "FP8 Formats for Deep Learning"
authors: "Micikevicius et al."
year: 2022
venue: "arXiv preprint"
arxiv: "2209.05433"
url: "https://arxiv.org/abs/2209.05433"
track: gpu-systems
tier: extended
est_hours: 3
status: stub
summary: "Defines E4M3 and E5M2 and explains when each fits forward vs. gradient tensors."
related_course: "gpu: numerics"
prereqs:
  - id: mixed-precision
    reason: "FP32 master copy of the weights"
background: [floating-point-formats, dynamic-range]
course_links:
  - title: "Floating point on the GPU"
    url: "https://gpu.ifkash.dev/c/09-floating-point-on-the-gpu"
  - title: "Quantized kernels"
    url: "https://gpu.ifkash.dev/c/18-quantized-kernels"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- E4M3 for weights and activations, E5M2 for gradients
- Per-tensor scaling
- Accuracy matches 16-bit on large models
