---
id: "decision-transformer"
title: "Decision Transformer: Reinforcement Learning via Sequence Modeling"
authors: "Chen et al."
year: 2021
venue: "NeurIPS 2021"
arxiv: "2106.01345"
url: "https://arxiv.org/abs/2106.01345"
track: rl
tier: core
est_hours: 5
status: stub
summary: "Recasts RL as conditional sequence modeling, bridging RL and LLM practice."
related_course: "rl: RL as sequence modeling"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
  - id: dqn
    reason: "Q-network trained on TD targets"
background: [offline-rl, return-to-go]
course_links:
  - title: "Offline reinforcement learning"
    url: "https://rl.ifkash.dev/lessons/045-offline-rl"
  - title: "Imitation learning"
    url: "https://rl.ifkash.dev/lessons/043-imitation-learning"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Condition on return-to-go
- Trajectory tokens (R, s, a)
- No bootstrapping or value function
