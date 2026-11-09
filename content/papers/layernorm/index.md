---
id: "layernorm"
title: "Layer Normalization"
authors: "Ba et al."
year: 2016
venue: "arXiv preprint"
arxiv: "1607.06450"
url: "https://arxiv.org/abs/1607.06450"
track: foundations
tier: core
est_hours: 2
status: stub
summary: "The normalization used inside every Transformer; understanding it explains why batch statistics are avoided for sequences."
related_course: "ml: normalization"
prereqs: []
background: [backprop, mean-and-variance, batch-statistics]
course_links:
  - title: "Normalisation layers: batch norm, layer norm, and their gradients"
    url: "https://ml.ifkash.dev/lessons/033-normalisation-layers"
  - title: "The transformer block: SwiGLU, residuals, and layer norm"
    url: "https://ml.ifkash.dev/lessons/050-the-transformer-block"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Normalize across features per example, not across the batch
- Learned gain and bias
- Independent of batch size, so it works for RNNs and autoregressive decoding
