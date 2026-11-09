---
id: "unet"
title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
authors: "Ronneberger et al."
year: 2015
venue: "MICCAI 2015"
arxiv: "1505.04597"
url: "https://arxiv.org/abs/1505.04597"
track: vision
tier: extended
est_hours: 3
status: stub
summary: "The backbone of early diffusion models."
related_course: "ml: CNNs"
prereqs:
  - id: resnet
    reason: "Learn residual F(x) instead of H(x)"
background: [convolution, downsampling-and-upsampling]
course_links:
  - title: "Convolution: slide a filter, count the weights, count the view"
    url: "https://ml.ifkash.dev/lessons/038-convolution-and-receptive-field"
  - title: "Diffusion for audio"
    url: "https://speech.ifkash.dev/lessons/055-diffusion-for-audio"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Contracting and expanding paths
- Skip connections between matching resolutions
- Dense per-pixel prediction
