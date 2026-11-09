---
id: "sglang"
title: "SGLang: Efficient Execution of Structured Language Model Programs"
authors: "Zheng et al."
year: 2023
venue: "NeurIPS 2024"
arxiv: "2312.07104"
url: "https://arxiv.org/abs/2312.07104"
track: inference
tier: extended
est_hours: 4
status: stub
summary: "RadixAttention automates KV-cache reuse across requests."
related_course: "inference: prefix caching"
prereqs:
  - id: pagedattention
    reason: "Fixed-size KV blocks with a block table"
background: [radix-tree, prefix-caching]
course_links:
  - title: "Paged attention"
    url: "https://qwen.ifkash.dev/c/15-paged-attention"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Radix tree over token prefixes
- LRU eviction of cached KV
- Compressed finite-state machines for constrained decoding
