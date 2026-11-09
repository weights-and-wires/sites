---
id: "muon"
title: "Muon: An optimizer for hidden layers in neural networks"
authors: "Jordan et al."
year: 2024
venue: "Blog post (Keller Jordan)"
arxiv: null
url: "https://kellerjordan.github.io/posts/muon/"
track: training-optimization
tier: core
est_hours: 3
status: stub
summary: "The optimizer behind several 2025 speed records and frontier training runs; a matrix-aware alternative to Adam."
related_course: "ml: optimizers"
prereqs:
  - id: adamw
    reason: "Weight decay applied directly to weights, not through the gradient"
background: [svd, spectral-norm, newton-schulz-iteration]
course_links:
  - title: "Lost digits, and how QR and the SVD keep them"
    url: "https://ml.ifkash.dev/lessons/004-stable-least-squares"
  - title: "SGD, Adam, and learning-rate schedules"
    url: "https://ml.ifkash.dev/lessons/031-sgd-adam-and-schedules"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Orthogonalize the momentum update of each weight matrix
- Newton-Schulz iteration approximates the orthogonalization cheaply
- Use Adam for embeddings and norms, Muon for hidden matrices
