---
id: "ldm"
title: "High-Resolution Image Synthesis with Latent Diffusion Models"
authors: "Rombach et al."
year: 2021
venue: "CVPR 2022"
arxiv: "2112.10752"
url: "https://arxiv.org/abs/2112.10752"
track: generative
tier: core
est_hours: 4
status: stub
summary: "Moves diffusion into a learned latent space; the basis of Stable Diffusion."
related_course: "ml: diffusion"
prereqs:
  - id: ddpm
    reason: "Fixed forward noising process with closed-form q(x_t|x_0)"
  - id: vae
    reason: "Evidence lower bound"
  - id: unet
    reason: "Contracting and expanding paths"
background: [autoencoders, cross-attention]
course_links:
  - title: "Diffusion for audio"
    url: "https://speech.ifkash.dev/lessons/055-diffusion-for-audio"
  - title: "Diffusion from scratch"
    url: "https://speech.ifkash.dev/lessons/052-diffusion-from-scratch"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Perceptual compression via an autoencoder
- Diffusion in latent space
- Cross-attention conditioning
