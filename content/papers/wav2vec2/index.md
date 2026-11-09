---
id: "wav2vec2"
title: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
authors: "Baevski et al."
year: 2020
venue: "NeurIPS 2020"
arxiv: "2006.11477"
url: "https://arxiv.org/abs/2006.11477"
track: speech
tier: core
est_hours: 5
status: stub
summary: "Self-supervised pretraining reduces labeled-data needs by orders of magnitude."
related_course: "speech: self-supervised speech"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
  - id: ctc
    reason: "Blank symbol and collapse mapping"
background: [contrastive-learning, vector-quantization, self-supervised-learning]
course_links:
  - title: "Self-supervised speech"
    url: "https://speech.ifkash.dev/lessons/036-self-supervised-speech"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- CNN feature encoder plus Transformer context network
- Quantized targets with a contrastive loss
- Fine-tune with CTC on small labeled sets
