---
id: "megatron"
title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
authors: "Shoeybi et al."
year: 2019
venue: "arXiv preprint"
arxiv: "1909.08053"
url: "https://arxiv.org/abs/1909.08053"
track: gpu-systems
tier: core
est_hours: 5
status: stub
summary: "Tensor parallelism splits matmuls across GPUs with minimal communication."
related_course: "gpu: distributed training"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [collective-communication, all-reduce, matrix-partitioning]
course_links:
  - title: "Tensor parallelism"
    url: "https://qwen.ifkash.dev/c/19-tensor-parallelism"
  - title: "Overlap and parallelism"
    url: "https://gpu.ifkash.dev/c/22-overlap-and-parallelism"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Column-parallel then row-parallel linear layers
- One all-reduce per block in each of forward and backward
- Split attention heads across devices
