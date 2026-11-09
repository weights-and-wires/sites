---
id: "cfg"
title: "Classifier-Free Diffusion Guidance"
authors: "Ho et al."
year: 2022
venue: "NeurIPS 2021 Workshop"
arxiv: "2207.12598"
url: "https://arxiv.org/abs/2207.12598"
track: generative
tier: extended
est_hours: 3
status: stub
summary: "The universal sampling trick for conditional diffusion and flow models."
related_course: "ml: diffusion"
prereqs:
  - id: ddpm
    reason: "Fixed forward noising process with closed-form q(x_t|x_0)"
background: [conditional-generation]
course_links:
  - title: "Classifier-free guidance"
    url: "https://speech.ifkash.dev/lessons/054-classifier-free-guidance"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Train conditional and unconditional jointly via condition dropout
- Extrapolate between predictions with a guidance scale
- Trades diversity for fidelity
