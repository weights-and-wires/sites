---
id: "grpo"
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
authors: "Shao et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2402.03300"
url: "https://arxiv.org/abs/2402.03300"
track: rl
tier: core
est_hours: 5
status: stub
summary: "Introduces GRPO, which drops the value network by normalizing rewards within a group of samples."
related_course: "rl: reasoning post-training"
prereqs:
  - id: ppo
    reason: "Clipped surrogate objective"
  - id: instructgpt
    reason: "SFT then reward model then PPO"
background: [group-relative-baselines, outcome-rewards]
course_links:
  - title: "Group-relative policy optimization"
    url: "https://rl.ifkash.dev/lessons/055-group-relative-policy-optimization"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Sample a group per prompt
- Advantage = standardized group reward
- KL as a direct loss term, not in the reward
