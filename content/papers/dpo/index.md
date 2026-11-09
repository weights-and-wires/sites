---
id: "dpo"
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
authors: "Rafailov et al."
year: 2023
venue: "NeurIPS 2023"
arxiv: "2305.18290"
url: "https://arxiv.org/abs/2305.18290"
track: rl
tier: core
est_hours: 5
status: stub
summary: "Skips the explicit reward model and RL loop by deriving a classification loss on preferences."
related_course: "rl: preference optimization"
prereqs:
  - id: instructgpt
    reason: "SFT then reward model then PPO"
background: [log-probability-ratios, closed-form-optimal-policy]
course_links:
  - title: "Direct preference optimization"
    url: "https://rl.ifkash.dev/lessons/054-direct-preference-optimization"
  - title: "PPO and DPO: two ways to reach the leashed optimum"
    url: "https://ml.ifkash.dev/lessons/059-ppo-and-dpo"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Optimal KL-regularized policy has a closed form
- Implicit reward = beta * log ratio
- Binary cross-entropy on preference pairs
