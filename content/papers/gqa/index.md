---
id: "gqa"
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
authors: "Ainslie et al."
year: 2023
venue: "EMNLP 2023"
arxiv: "2305.13245"
url: "https://arxiv.org/abs/2305.13245"
track: architectures
tier: core
est_hours: 3
status: stub
summary: "Shrinks the KV cache by sharing key/value heads, a direct lever on serving cost."
related_course: "inference: KV cache"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [kv-cache, memory-bandwidth]
course_links:
  - title: "Grouped-query attention"
    url: "https://qwen.ifkash.dev/c/07-grouped-query-attention"
  - title: "The KV cache"
    url: "https://qwen.ifkash.dev/c/09-the-kv-cache"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Groups of query heads share one KV head
- Interpolates between multi-head and multi-query attention
- Uptrain from an existing checkpoint
