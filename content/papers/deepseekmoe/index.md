---
id: "deepseekmoe"
title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
authors: "Dai et al."
year: 2024
venue: "ACL 2024"
arxiv: "2401.06066"
url: "https://arxiv.org/abs/2401.06066"
track: scaling-efficiency
tier: extended
est_hours: 3
status: stub
summary: "Fine-grained and shared experts, the MoE design reused in DeepSeek-V2 and V3."
related_course: "ml: mixture of experts"
prereqs:
  - id: mixtral
    reason: "8 experts, top-2 routing per token"
background: [expert-specialization]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/060-mixtures-of-experts"
  - title: "Mixture-of-experts and sparse kernels"
    url: "https://gpu.ifkash.dev/c/20-moe-and-sparse-kernels"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Fine-grained expert segmentation
- Shared experts capture common knowledge
- Device-limited routing
