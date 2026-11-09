---
id: "dit"
title: "Scalable Diffusion Models with Transformers"
authors: "Peebles et al."
year: 2022
venue: "ICCV 2023"
arxiv: "2212.09748"
url: "https://arxiv.org/abs/2212.09748"
track: generative
tier: core
est_hours: 4
status: stub
summary: "Replaces the U-Net with a Transformer and shows clean scaling; the backbone of current image and video models."
related_course: "ml: diffusion"
prereqs:
  - id: ldm
    reason: "Perceptual compression via an autoencoder"
  - id: vit
    reason: "Split image into patches and linearly embed"
  - id: cfg
    reason: "Train conditional and unconditional jointly via condition dropout"
background: [adaln-conditioning, patchify]
course_links:
  - title: "Diffusion for audio"
    url: "https://speech.ifkash.dev/lessons/055-diffusion-for-audio"
  - title: "Diffusion from scratch"
    url: "https://speech.ifkash.dev/lessons/052-diffusion-from-scratch"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Latent patches as tokens
- adaLN-Zero conditioning on timestep and class
- Gflops predicts FID
