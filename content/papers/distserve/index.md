---
id: "distserve"
title: "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"
authors: "Zhong et al."
year: 2024
venue: "OSDI 2024"
arxiv: "2401.09670"
url: "https://arxiv.org/abs/2401.09670"
track: inference
tier: extended
est_hours: 4
status: stub
summary: "Separates prefill and decode onto different GPUs to avoid interference."
related_course: "inference: serving architecture"
prereqs:
  - id: sarathi
    reason: "Split prefill into chunks"
background: [goodput, slos]
course_links:
  - title: "Benchmarks that mean something"
    url: "https://qwen.ifkash.dev/c/17-benchmarks"
  - title: "Continuous batching"
    url: "https://qwen.ifkash.dev/c/16-continuous-batching"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Prefill-decode interference
- Per-phase parallelism and placement
- Goodput under TTFT/TPOT SLOs
