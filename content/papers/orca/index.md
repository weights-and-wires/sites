---
id: "orca"
title: "Orca: A Distributed Serving System for Transformer-Based Generative Models"
authors: "Yu et al."
year: 2022
venue: "OSDI 2022"
arxiv: null
url: "https://www.usenix.org/conference/osdi22/presentation/yu"
track: inference
tier: core
est_hours: 4
status: stub
summary: "Introduces iteration-level scheduling, known today as continuous batching."
related_course: "inference: continuous batching"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [batching, autoregressive-decoding, scheduling]
course_links:
  - title: "Continuous batching"
    url: "https://qwen.ifkash.dev/c/16-continuous-batching"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Schedule at each decoding iteration, not each request
- Selective batching for attention
- Reduces head-of-line blocking
