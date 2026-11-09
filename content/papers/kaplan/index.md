---
id: "kaplan"
title: "Scaling Laws for Neural Language Models"
authors: "Kaplan et al."
year: 2020
venue: "arXiv preprint"
arxiv: "2001.08361"
url: "https://arxiv.org/abs/2001.08361"
track: scaling-efficiency
tier: core
est_hours: 3
status: stub
summary: "Establishes that loss follows power laws in parameters, data, and compute."
related_course: "ml: scaling laws"
prereqs:
  - id: gpt2
    reason: "Decoder-only Transformer with pre-LN"
background: [power-laws, log-log-regression]
course_links:
  - title: "Scaling laws and the compute budget"
    url: "https://ml.ifkash.dev/lessons/055-scaling-laws"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Loss vs. N, D, C power laws
- Larger models are more sample-efficient
- Compute-optimal allocation (later revised)
