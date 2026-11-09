---
id: "chinchilla"
title: "Training Compute-Optimal Large Language Models"
authors: "Hoffmann et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2203.15556"
url: "https://arxiv.org/abs/2203.15556"
track: scaling-efficiency
tier: core
est_hours: 3
status: stub
summary: "Corrects Kaplan: parameters and tokens should scale together, roughly 20 tokens per parameter."
related_course: "ml: scaling laws"
prereqs:
  - id: kaplan
    reason: "Loss vs. N, D, C power laws"
background: [flops-estimate-6nd, curve-fitting]
course_links:
  - title: "Scaling laws and the compute budget"
    url: "https://ml.ifkash.dev/lessons/055-scaling-laws"
  - title: "The training step, and what a run costs"
    url: "https://ml.ifkash.dev/lessons/053-training-loop-and-cost"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Three estimation approaches agree on equal scaling of N and D
- Chinchilla 70B beats Gopher 280B at equal compute
- Basis for inference-aware overtraining later
