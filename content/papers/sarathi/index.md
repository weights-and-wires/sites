---
id: "sarathi"
title: "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"
authors: "Agrawal et al."
year: 2024
venue: "OSDI 2024"
arxiv: "2403.02310"
url: "https://arxiv.org/abs/2403.02310"
track: inference
tier: extended
est_hours: 4
status: stub
summary: "Chunked prefill and stall-free batching; the default scheduler idea in modern engines."
related_course: "inference: scheduling"
prereqs:
  - id: pagedattention
    reason: "Fixed-size KV blocks with a block table"
  - id: roofline
    reason: "Attainable FLOP/s = min(peak, bandwidth x intensity)"
background: [prefill-vs-decode, stall-free-scheduling]
course_links:
  - title: "Continuous batching"
    url: "https://qwen.ifkash.dev/c/16-continuous-batching"
  - title: "Benchmarks that mean something"
    url: "https://qwen.ifkash.dev/c/17-benchmarks"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Split prefill into chunks
- Piggyback decode on prefill iterations
- Token budget per iteration
