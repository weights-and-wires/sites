---
id: "adam"
title: "Adam: A Method for Stochastic Optimization"
authors: "Kingma et al."
year: 2014
venue: "ICLR 2015"
arxiv: "1412.6980"
url: "https://arxiv.org/abs/1412.6980"
track: training-optimization
tier: core
est_hours: 2
status: stub
summary: "The default optimizer for deep learning; every later optimizer is described relative to it."
related_course: "ml: optimizers"
prereqs: []
background: [sgd, momentum, exponential-moving-averages]
course_links:
  - title: "SGD, Adam, and learning-rate schedules"
    url: "https://ml.ifkash.dev/lessons/031-sgd-adam-and-schedules"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- First and second moment estimates per parameter
- Bias correction for the zero-initialized moving averages
- Per-parameter step size is roughly lr * sign-like normalized gradient
