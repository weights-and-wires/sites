---
id: "adamw"
title: "Decoupled Weight Decay Regularization"
authors: "Loshchilov et al."
year: 2017
venue: "ICLR 2019"
arxiv: "1711.05101"
url: "https://arxiv.org/abs/1711.05101"
track: training-optimization
tier: core
est_hours: 2
status: stub
summary: "Explains why L2 regularization and weight decay differ under Adam, which is why AdamW is the LLM default."
related_course: "ml: optimizers"
prereqs:
  - id: adam
    reason: "First and second moment estimates per parameter"
background: [l2-regularization, weight-decay]
course_links:
  - title: "Weight decay and early stopping: two ways to keep weights small"
    url: "https://ml.ifkash.dev/lessons/034-weight-decay-and-early-stopping"
  - title: "SGD, Adam, and learning-rate schedules"
    url: "https://ml.ifkash.dev/lessons/031-sgd-adam-and-schedules"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Weight decay applied directly to weights, not through the gradient
- Decoupling makes the optimal decay independent of learning rate
- Standard recipe in nearly all LLM training
