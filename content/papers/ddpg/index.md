---
id: "ddpg"
title: "Continuous control with deep reinforcement learning"
authors: "Lillicrap et al."
year: 2015
venue: "ICLR 2016"
arxiv: "1509.02971"
url: "https://arxiv.org/abs/1509.02971"
track: rl
tier: extended
est_hours: 4
status: stub
summary: "Extends DQN-style off-policy learning to continuous actions."
related_course: "rl: continuous control"
prereqs:
  - id: dqn
    reason: "Q-network trained on TD targets"
background: [deterministic-policy-gradient, ornstein-uhlenbeck-noise]
course_links:
  - title: "Deterministic policy gradients: DDPG and TD3"
    url: "https://rl.ifkash.dev/lessons/037-deterministic-policy-gradients"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Deterministic actor and Q critic
- Target networks with soft updates
- Exploration noise added to actions
