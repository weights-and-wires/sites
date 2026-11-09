---
id: "trpo"
title: "Trust Region Policy Optimization"
authors: "Schulman et al."
year: 2015
venue: "ICML 2015"
arxiv: "1502.05477"
url: "https://arxiv.org/abs/1502.05477"
track: rl
tier: core
est_hours: 6
status: stub
summary: "Motivates why policy updates need constraints; PPO simplifies it."
related_course: "rl: policy gradients"
prereqs: []
background: [policy-gradient-theorem, kl-divergence, conjugate-gradient]
course_links:
  - title: "Natural gradients and trust regions"
    url: "https://rl.ifkash.dev/lessons/033-trust-regions-and-trpo"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Surrogate objective with monotonic improvement bound
- KL trust region on policy change
- Natural gradient step via conjugate gradient
