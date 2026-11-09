---
id: "r1"
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors: "DeepSeek-AI et al."
year: 2025
venue: "arXiv preprint (Nature 2025)"
arxiv: "2501.12948"
url: "https://arxiv.org/abs/2501.12948"
track: rl
tier: core
est_hours: 6
status: stub
summary: "Shows reasoning behavior emerging from RL with rule-based rewards, plus a practical multi-stage recipe."
related_course: "rl: reasoning models"
prereqs:
  - id: grpo
    reason: "Sample a group per prompt"
  - id: cot
    reason: "Include reasoning steps in the exemplars"
background: [verifiable-rewards, distillation]
course_links:
  - title: "Reasoning with verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/057-reasoning-with-verifiable-rewards"
  - title: "Verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/056-verifiable-rewards"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- R1-Zero: pure RL with accuracy and format rewards
- Cold-start SFT, then RL, rejection sampling, final RL
- Distill into small models
