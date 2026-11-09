---
id: "sac"
title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
authors: "Haarnoja et al."
year: 2018
venue: "ICML 2018"
arxiv: "1801.01290"
url: "https://arxiv.org/abs/1801.01290"
track: rl
tier: extended
est_hours: 5
status: stub
summary: "Maximum-entropy RL that is stable and sample-efficient; the standard off-policy baseline."
related_course: "rl: continuous control"
prereqs:
  - id: ddpg
    reason: "Deterministic actor and Q critic"
background: [entropy-regularization, reparameterization-trick]
course_links:
  - title: "Soft actor-critic"
    url: "https://rl.ifkash.dev/lessons/039-soft-actor-critic"
  - title: "Maximum-entropy reinforcement learning"
    url: "https://rl.ifkash.dev/lessons/038-maximum-entropy-rl"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Maximize reward plus policy entropy
- Twin Q critics
- Reparameterized stochastic policy
