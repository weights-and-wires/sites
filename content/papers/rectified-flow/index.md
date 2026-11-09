---
id: "rectified-flow"
title: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
authors: "Liu et al."
year: 2022
venue: "ICLR 2023"
arxiv: "2209.03003"
url: "https://arxiv.org/abs/2209.03003"
track: generative
tier: core
est_hours: 4
status: stub
summary: "Linear interpolation paths plus reflow; the training objective behind Stable Diffusion 3 and similar."
related_course: "ml: flow models"
prereqs:
  - id: flow-matching
    reason: "Regress a conditional vector field"
background: [optimal-transport-intuition]
course_links:
  - title: "Straighter paths and better solvers"
    url: "https://speech.ifkash.dev/lessons/057-straighter-paths-and-better-solvers"
  - title: "Flow matching"
    url: "https://speech.ifkash.dev/lessons/056-flow-matching"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Straight-line interpolation between noise and data
- Reflow straightens the learned trajectories
- Few-step sampling
