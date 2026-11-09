---
id: "spec-decoding"
title: "Fast Inference from Transformers via Speculative Decoding"
authors: "Leviathan et al."
year: 2022
venue: "ICML 2023"
arxiv: "2211.17192"
url: "https://arxiv.org/abs/2211.17192"
track: inference
tier: core
est_hours: 4
status: stub
summary: "Lossless speedup by verifying draft-model tokens in parallel."
related_course: "inference: speculative decoding"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [rejection-sampling, autoregressive-decoding]
course_links:
  - title: "Speculative decoding"
    url: "https://qwen.ifkash.dev/c/20-speculative-decoding"
  - title: "Decoding and sampling: cache, temperature, speculation"
    url: "https://ml.ifkash.dev/lessons/056-decoding-and-sampling"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Draft k tokens cheaply, verify with one target pass
- Rejection sampling preserves the target distribution
- Speedup depends on acceptance rate
