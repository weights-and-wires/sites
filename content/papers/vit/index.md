---
id: "vit"
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: "Dosovitskiy et al."
year: 2020
venue: "ICLR 2021"
arxiv: "2010.11929"
url: "https://arxiv.org/abs/2010.11929"
track: vision
tier: core
est_hours: 4
status: stub
summary: "Shows pure Transformers work on images, which enabled DiT and multimodal LLMs."
related_course: "ml: vision Transformers"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [patch-embedding, image-classification]
course_links:
  - title: "Scaled attention, many heads, and the causal mask"
    url: "https://ml.ifkash.dev/lessons/047-scaled-attention-and-heads"
  - title: "Convolution: slide a filter, count the weights, count the view"
    url: "https://ml.ifkash.dev/lessons/038-convolution-and-receptive-field"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Split image into patches and linearly embed
- Class token and learned position embeddings
- Needs large-scale pretraining
