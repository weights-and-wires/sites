---
id: "llama"
title: "LLaMA: Open and Efficient Foundation Language Models"
authors: "Touvron et al."
year: 2023
venue: "arXiv preprint"
arxiv: "2302.13971"
url: "https://arxiv.org/abs/2302.13971"
track: nlp-llm
tier: core
est_hours: 4
status: stub
summary: "The reference open architecture: RMSNorm, SwiGLU, RoPE, trained far past Chinchilla-optimal."
related_course: "ml: LLM architecture"
prereqs:
  - id: gpt3
    reason: "Scale to 175B parameters"
  - id: rope
    reason: "Rotate query and key pairs by position-dependent angles"
  - id: layernorm
    reason: "Normalize across features per example, not across the batch"
  - id: chinchilla
    reason: "Three estimation approaches agree on equal scaling of N and D"
background: [rmsnorm, swiglu]
course_links:
  - title: "The transformer block: SwiGLU, residuals, and layer norm"
    url: "https://ml.ifkash.dev/lessons/050-the-transformer-block"
  - title: "RMSNorm and the residual stream"
    url: "https://qwen.ifkash.dev/c/04-rmsnorm-and-residuals"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Pre-norm with RMSNorm
- SwiGLU feed-forward
- Train smaller models on more tokens
