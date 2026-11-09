---
id: "snell"
title: "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters"
authors: "Snell et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2408.03314"
url: "https://arxiv.org/abs/2408.03314"
track: scaling-efficiency
tier: extended
est_hours: 3
status: stub
summary: "Frames inference-time compute as a scaling axis traded against model size."
related_course: "rl: test-time compute"
prereqs:
  - id: prm
    reason: "Step-level human labels (PRM800K)"
  - id: cot
    reason: "Include reasoning steps in the exemplars"
background: [best-of-n, beam-search, compute-optimal-allocation]
course_links:
  - title: "Process rewards and test-time search"
    url: "https://rl.ifkash.dev/lessons/058-process-rewards-and-search"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Search against a PRM and sequential revisions
- Difficulty-adaptive compute allocation
- Small model plus test compute can beat a larger model
