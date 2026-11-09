---
id: "ppo"
title: "Proximal Policy Optimization Algorithms"
authors: "Schulman et al."
year: 2017
venue: "arXiv preprint"
arxiv: "1707.06347"
url: "https://arxiv.org/abs/1707.06347"
track: rl
tier: core
est_hours: 6
status: stub
summary: "The workhorse algorithm of deep RL and RLHF."
related_course: "rl: policy gradients"
prereqs:
  - id: trpo
    reason: "Surrogate objective with monotonic improvement bound"
  - id: gae
    reason: "Exponentially weighted sum of TD residuals"
background: [importance-sampling, clipping]
course_links:
  - title: "Proximal policy optimization"
    url: "https://rl.ifkash.dev/lessons/034-proximal-policy-optimization"
  - title: "PPO in practice"
    url: "https://rl.ifkash.dev/lessons/035-ppo-in-practice"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Clipped surrogate objective
- Multiple epochs over the same batch
- Value and entropy terms in one loss
