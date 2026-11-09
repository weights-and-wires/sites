---
id: "zero"
title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
authors: "Rajbhandari et al."
year: 2019
venue: "SC 2020"
arxiv: "1910.02054"
url: "https://arxiv.org/abs/1910.02054"
track: gpu-systems
tier: core
est_hours: 4
status: stub
summary: "Shards optimizer state, gradients, and parameters; the basis of FSDP and DeepSpeed."
related_course: "gpu: distributed training"
prereqs:
  - id: megatron
    reason: "Column-parallel then row-parallel linear layers"
  - id: adam
    reason: "First and second moment estimates per parameter"
  - id: mixed-precision
    reason: "FP32 master copy of the weights"
background: [data-parallelism, optimizer-state-memory]
course_links:
  - title: "Data parallel and ZeRO-1"
    url: "https://gpu.ifkash.dev/c/25-ddp-and-zero"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Optimizer state dominates memory (12 bytes/param for Adam mixed precision)
- Stage 1, 2, 3 shard progressively more
- Same communication as data parallel for stages 1-2
