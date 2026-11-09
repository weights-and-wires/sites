---
id: "clip"
title: "Learning Transferable Visual Models From Natural Language Supervision"
authors: "Radford et al."
year: 2021
venue: "ICML 2021"
arxiv: "2103.00020"
url: "https://arxiv.org/abs/2103.00020"
track: vision
tier: core
est_hours: 3
status: stub
summary: "Contrastive image-text pretraining powers text conditioning for diffusion and vision-language models."
related_course: "ml: multimodal"
prereqs:
  - id: vit
    reason: "Split image into patches and linearly embed"
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [contrastive-learning, infonce]
course_links:
  - title: "Self-supervised speech"
    url: "https://speech.ifkash.dev/lessons/036-self-supervised-speech"
  - title: "Attention as a soft lookup, with a dot product or a learned scorer"
    url: "https://ml.ifkash.dev/lessons/045-attention-as-soft-lookup"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Symmetric contrastive loss over a batch
- Zero-shot classification by prompt
- 400M image-text pairs
