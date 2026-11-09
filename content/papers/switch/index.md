---
id: "switch"
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
authors: "Fedus et al."
year: 2021
venue: "JMLR 2022"
arxiv: "2101.03961"
url: "https://arxiv.org/abs/2101.03961"
track: scaling-efficiency
tier: extended
est_hours: 4
status: stub
summary: "Simplifies MoE to top-1 routing and makes sparse scaling practical."
related_course: "ml: mixture of experts"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [top-k-routing, load-balancing]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/060-mixtures-of-experts"
  - title: "Mixture-of-experts and sparse kernels"
    url: "https://gpu.ifkash.dev/c/20-moe-and-sparse-kernels"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Top-1 expert routing
- Auxiliary load-balancing loss
- Capacity factor and token dropping
