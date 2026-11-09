---
id: "fa3"
title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
authors: "Shah et al."
year: 2024
venue: "NeurIPS 2024"
arxiv: "2407.08608"
url: "https://arxiv.org/abs/2407.08608"
track: gpu-systems
tier: extended
est_hours: 5
status: stub
summary: "Exploits Hopper asynchrony and FP8; shows how kernel design tracks hardware generations."
related_course: "gpu: Hopper kernels"
prereqs:
  - id: fa2
    reason: "Reduce non-matmul FLOPs"
  - id: fp8
    reason: "E4M3 for weights and activations, E5M2 for gradients"
background: [hopper-tma-and-wgmma, warp-specialization]
course_links:
  - title: "FlashAttention"
    url: "https://gpu.ifkash.dev/c/16-flash-attention"
  - title: "Hopper GEMM"
    url: "https://gpu.ifkash.dev/c/26-hopper-gemm"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Warp-specialized producer-consumer pipeline
- Overlap softmax with GEMMs
- FP8 with block quantization and incoherent processing
