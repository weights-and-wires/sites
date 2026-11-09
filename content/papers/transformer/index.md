---
id: "transformer"
title: "Attention Is All You Need"
authors: "Vaswani et al."
year: 2017
venue: "NeurIPS 2017"
arxiv: "1706.03762"
url: "https://arxiv.org/abs/1706.03762"
track: architectures
tier: core
est_hours: 6
status: stub
summary: "The architecture behind every modern LLM; the single most connected node in the graph."
related_course: "ml: transformers"
prereqs:
  - id: bahdanau
    reason: "Additive (MLP) alignment scores"
  - id: layernorm
    reason: "Normalize across features per example, not across the batch"
  - id: resnet
    reason: "Learn residual F(x) instead of H(x)"
background: [matrix-multiplication, softmax-numerics, residual-connections, positional-information]
course_links:
  - title: "Scaled attention, many heads, and the causal mask"
    url: "https://ml.ifkash.dev/lessons/047-scaled-attention-and-heads"
  - title: "The transformer block: SwiGLU, residuals, and layer norm"
    url: "https://ml.ifkash.dev/lessons/050-the-transformer-block"
  - title: "Position encodings: sinusoidal clocks and rotary embeddings"
    url: "https://ml.ifkash.dev/lessons/049-position-encodings"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Scaled dot-product self-attention
- Multi-head attention
- Positional encodings plus residual and LayerNorm blocks
