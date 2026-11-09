---
id: "pagedattention"
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
authors: "Kwon et al."
year: 2023
venue: "SOSP 2023"
arxiv: "2309.06180"
url: "https://arxiv.org/abs/2309.06180"
track: inference
tier: core
est_hours: 5
status: stub
summary: "Applies OS paging to the KV cache; the core idea in vLLM."
related_course: "inference: vLLM"
prereqs:
  - id: orca
    reason: "Schedule at each decoding iteration, not each request"
background: [kv-cache, virtual-memory-and-paging, fragmentation]
course_links:
  - title: "Paged attention"
    url: "https://qwen.ifkash.dev/c/15-paged-attention"
  - title: "The KV cache"
    url: "https://qwen.ifkash.dev/c/09-the-kv-cache"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Fixed-size KV blocks with a block table
- Near-zero fragmentation
- Copy-on-write sharing for beam search and prefixes
