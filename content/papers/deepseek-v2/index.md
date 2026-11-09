---
id: "deepseek-v2"
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
authors: "DeepSeek-AI et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2405.04434"
url: "https://arxiv.org/abs/2405.04434"
track: architectures
tier: core
est_hours: 5
status: stub
summary: "Introduces Multi-head Latent Attention (MLA), which compresses the KV cache far below GQA."
related_course: "inference: KV cache compression"
prereqs:
  - id: gqa
    reason: "Groups of query heads share one KV head"
  - id: deepseekmoe
    reason: "Fine-grained expert segmentation"
  - id: rope
    reason: "Rotate query and key pairs by position-dependent angles"
background: [low-rank-projection, kv-cache]
course_links:
  - title: "The KV cache"
    url: "https://qwen.ifkash.dev/c/09-the-kv-cache"
  - title: "Grouped-query attention"
    url: "https://qwen.ifkash.dev/c/07-grouped-query-attention"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Latent KV compression with up-projection
- Decoupled RoPE for compatibility
- DeepSeekMoE for the FFN
