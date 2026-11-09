---
id: "mamba"
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
authors: "Gu et al."
year: 2023
venue: "COLM 2024"
arxiv: "2312.00752"
url: "https://arxiv.org/abs/2312.00752"
track: architectures
tier: core
est_hours: 5
status: stub
summary: "The leading non-attention sequence model; linear-time inference with a constant-size state."
related_course: "ml: alternatives to attention"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
  - id: lstm
    reason: "Additive cell state avoids vanishing gradients"
background: [linear-recurrences, parallel-scan, state-space-models]
course_links:
  - title: "Linear attention and the gated delta rule"
    url: "https://qwen.ifkash.dev/c/06-gated-delta-rule"
  - title: "Scan and histogram"
    url: "https://gpu.ifkash.dev/c/07-scan-and-histogram"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Input-dependent (selective) SSM parameters
- Hardware-aware parallel scan
- Constant memory per token at inference
