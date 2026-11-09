---
id: "rope"
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors: "Su et al."
year: 2021
venue: "Neurocomputing 2024"
arxiv: "2104.09864"
url: "https://arxiv.org/abs/2104.09864"
track: architectures
tier: core
est_hours: 3
status: stub
summary: "The position encoding in almost every open LLM; the basis for long-context extension methods."
related_course: "ml: positional encoding"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [complex-numbers-and-rotations, dot-products]
course_links:
  - title: "Position encodings: sinusoidal clocks and rotary embeddings"
    url: "https://ml.ifkash.dev/lessons/049-position-encodings"
  - title: "Rotary position embeddings"
    url: "https://qwen.ifkash.dev/c/05-rotary-embeddings"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Rotate query and key pairs by position-dependent angles
- Dot product depends only on relative offset
- Extrapolation methods rescale the angles
