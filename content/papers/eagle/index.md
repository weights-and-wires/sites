---
id: "eagle"
title: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
authors: "Li et al."
year: 2024
venue: "ICML 2024"
arxiv: "2401.15077"
url: "https://arxiv.org/abs/2401.15077"
track: inference
tier: extended
est_hours: 4
status: stub
summary: "State-of-the-art lossless drafting using the target model's features."
related_course: "inference: speculative decoding"
prereqs:
  - id: spec-decoding
    reason: "Draft k tokens cheaply, verify with one target pass"
background: [feature-level-drafting, tree-attention]
course_links:
  - title: "Speculative decoding"
    url: "https://qwen.ifkash.dev/c/20-speculative-decoding"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Autoregress on the second-to-top-layer features
- Resolve uncertainty with the advanced token
- Tree-structured drafts
