---
id: "mixtral"
title: "Mixtral of Experts"
authors: "Jiang et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2401.04088"
url: "https://arxiv.org/abs/2401.04088"
track: scaling-efficiency
tier: core
est_hours: 3
status: stub
summary: "A clean open reference for sparse MoE LLMs; shows active-parameter economics."
related_course: "inference: MoE serving"
prereqs:
  - id: switch
    reason: "Top-1 expert routing"
  - id: llama
    reason: "Pre-norm with RMSNorm"
background: [top-2-routing, active-vs-total-parameters]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/060-mixtures-of-experts"
  - title: "Mixture-of-experts and sparse kernels"
    url: "https://gpu.ifkash.dev/c/20-moe-and-sparse-kernels"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- 8 experts, top-2 routing per token
- 47B total vs. 13B active parameters
- Router specialization is syntactic more than topical
