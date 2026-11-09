---
id: "flashattention"
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors: "Dao et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2205.14135"
url: "https://arxiv.org/abs/2205.14135"
track: gpu-systems
tier: core
est_hours: 6
status: stub
summary: "Turns attention from memory-bound to compute-bound by tiling in SRAM; the canonical IO-aware algorithm."
related_course: "gpu: attention kernels"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
  - id: roofline
    reason: "Attainable FLOP/s = min(peak, bandwidth x intensity)"
  - id: online-softmax
    reason: "Maintain running max and running sum"
background: [gpu-memory-hierarchy, sram-vs-hbm, tiling, recomputation]
course_links:
  - title: "FlashAttention"
    url: "https://gpu.ifkash.dev/c/16-flash-attention"
  - title: "FlashAttention"
    url: "https://qwen.ifkash.dev/c/14-flash-attention"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Tile Q, K, V and fuse softmax with the matmuls
- Never materialize the n x n matrix
- Recompute in the backward pass instead of storing
