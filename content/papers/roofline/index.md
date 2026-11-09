---
id: "roofline"
title: "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
authors: "Williams et al."
year: 2009
venue: "Communications of the ACM 52(4)"
arxiv: null
url: "https://doi.org/10.1145/1498765.1498785"
track: gpu-systems
tier: core
est_hours: 3
status: stub
summary: "The mental model for deciding whether a kernel is compute- or memory-bound; used in every later systems paper."
related_course: "gpu: performance model"
prereqs: []
background: [arithmetic-intensity, memory-bandwidth, flops]
course_links:
  - title: "The roofline and latency hiding"
    url: "https://gpu.ifkash.dev/c/04-roofline-and-latency"
  - title: "The roofline"
    url: "https://qwen.ifkash.dev/c/10-roofline"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Attainable FLOP/s = min(peak, bandwidth x intensity)
- Ridge point separates memory- and compute-bound regimes
- Optimizations move a kernel along or up the roof
