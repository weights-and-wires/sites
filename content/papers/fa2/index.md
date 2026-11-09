---
id: "fa2"
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
authors: "Dao et al."
year: 2023
venue: "ICLR 2024"
arxiv: "2307.08691"
url: "https://arxiv.org/abs/2307.08691"
track: gpu-systems
tier: core
est_hours: 4
status: stub
summary: "Improves FlashAttention's utilization by repartitioning work across thread blocks and warps."
related_course: "gpu: attention kernels"
prereqs:
  - id: flashattention
    reason: "Tile Q, K, V and fuse softmax with the matmuls"
background: [warps, occupancy, non-matmul-flops]
course_links:
  - title: "FlashAttention"
    url: "https://gpu.ifkash.dev/c/16-flash-attention"
  - title: "FlashAttention"
    url: "https://qwen.ifkash.dev/c/14-flash-attention"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Reduce non-matmul FLOPs
- Parallelize over sequence length
- Split work between warps to cut shared-memory traffic
