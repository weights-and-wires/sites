---
id: "score-sde"
title: "Score-Based Generative Modeling through Stochastic Differential Equations"
authors: "Song et al."
year: 2020
venue: "ICLR 2021"
arxiv: "2011.13456"
url: "https://arxiv.org/abs/2011.13456"
track: generative
tier: core
est_hours: 6
status: stub
summary: "Unifies diffusion and score matching as SDEs and introduces the probability-flow ODE."
related_course: "ml: diffusion"
prereqs:
  - id: ddpm
    reason: "Fixed forward noising process with closed-form q(x_t|x_0)"
background: [sdes, score-function, fokker-planck]
course_links:
  - title: "Sampling faster: DDIM and the probability-flow ODE"
    url: "https://speech.ifkash.dev/lessons/053-fast-samplers"
  - title: "Diffusion from scratch"
    url: "https://speech.ifkash.dev/lessons/052-diffusion-from-scratch"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Forward and reverse-time SDEs
- Score matching trains the reverse drift
- Probability-flow ODE gives deterministic sampling
