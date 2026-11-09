---
id: "flow-matching"
title: "Flow Matching for Generative Modeling"
authors: "Lipman et al."
year: 2022
venue: "ICLR 2023"
arxiv: "2210.02747"
url: "https://arxiv.org/abs/2210.02747"
track: generative
tier: core
est_hours: 5
status: stub
summary: "Simulation-free training of continuous flows with straight-ish paths; the core of modern generative recipes."
related_course: "ml: flow models"
prereqs:
  - id: score-sde
    reason: "Forward and reverse-time SDEs"
background: [odes, continuous-normalizing-flows, probability-paths]
course_links:
  - title: "Flow matching"
    url: "https://speech.ifkash.dev/lessons/056-flow-matching"
  - title: "Straighter paths and better solvers"
    url: "https://speech.ifkash.dev/lessons/057-straighter-paths-and-better-solvers"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Regress a conditional vector field
- Gaussian probability paths include diffusion as a special case
- Optimal-transport paths are straighter
