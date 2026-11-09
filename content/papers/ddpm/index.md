---
id: "ddpm"
title: "Denoising Diffusion Probabilistic Models"
authors: "Ho et al."
year: 2020
venue: "NeurIPS 2020"
arxiv: "2006.11239"
url: "https://arxiv.org/abs/2006.11239"
track: generative
tier: core
est_hours: 6
status: stub
summary: "The modern formulation of diffusion; start of the generative chain through flow matching."
related_course: "ml: diffusion"
prereqs:
  - id: vae
    reason: "Evidence lower bound"
  - id: unet
    reason: "Contracting and expanding paths"
background: [gaussian-noise, markov-chains, kl-divergence]
course_links:
  - title: "Diffusion from scratch"
    url: "https://speech.ifkash.dev/lessons/052-diffusion-from-scratch"
  - title: "Sampling faster: DDIM and the probability-flow ODE"
    url: "https://speech.ifkash.dev/lessons/053-fast-samplers"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Fixed forward noising process with closed-form q(x_t|x_0)
- Predict the noise epsilon
- Simplified loss from the ELBO
