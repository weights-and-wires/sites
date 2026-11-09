---
id: "gae"
title: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
authors: "Schulman et al."
year: 2015
venue: "ICLR 2016"
arxiv: "1506.02438"
url: "https://arxiv.org/abs/1506.02438"
track: rl
tier: core
est_hours: 3
status: stub
summary: "The advantage estimator used in PPO and RLHF implementations."
related_course: "rl: advantage estimation"
prereqs:
  - id: trpo
    reason: "Surrogate objective with monotonic improvement bound"
background: [td-lambda, bias-variance-trade-off, value-function]
course_links:
  - title: "Generalized advantage estimation"
    url: "https://rl.ifkash.dev/lessons/032-generalized-advantage-estimation"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Exponentially weighted sum of TD residuals
- Lambda trades bias for variance
- Reduces to TD(0) and Monte Carlo as special cases
