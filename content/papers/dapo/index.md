---
id: "dapo"
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
authors: "Yu et al."
year: 2025
venue: "arXiv preprint"
arxiv: "2503.14476"
url: "https://arxiv.org/abs/2503.14476"
track: rl
tier: extended
est_hours: 4
status: stub
summary: "Documents the fixes (clip-higher, dynamic sampling, token-level loss) that make GRPO-style training stable."
related_course: "rl: reasoning models"
prereqs:
  - id: r1
    reason: "R1-Zero: pure RL with accuracy and format rewards"
background: [entropy-collapse, dynamic-sampling]
course_links:
  - title: "Reasoning with verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/057-reasoning-with-verifiable-rewards"
  - title: "Group-relative policy optimization"
    url: "https://rl.ifkash.dev/lessons/055-group-relative-policy-optimization"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Decoupled clip ranges
- Filter prompts with all-correct or all-wrong groups
- Token-level loss and overlong reward shaping
