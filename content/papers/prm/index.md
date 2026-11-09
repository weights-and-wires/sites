---
id: "prm"
title: "Let's Verify Step by Step"
authors: "Lightman et al."
year: 2023
venue: "ICLR 2024"
arxiv: "2305.20050"
url: "https://arxiv.org/abs/2305.20050"
track: rl
tier: extended
est_hours: 3
status: stub
summary: "Process supervision beats outcome supervision for verifying math reasoning."
related_course: "rl: reasoning models"
prereqs:
  - id: instructgpt
    reason: "SFT then reward model then PPO"
  - id: cot
    reason: "Include reasoning steps in the exemplars"
background: [process-vs-outcome-supervision, best-of-n]
course_links:
  - title: "Process rewards and test-time search"
    url: "https://rl.ifkash.dev/lessons/058-process-rewards-and-search"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Step-level human labels (PRM800K)
- Best-of-N reranking with a PRM
- Active learning for labeling
