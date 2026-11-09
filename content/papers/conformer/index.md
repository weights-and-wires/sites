---
id: "conformer"
title: "Conformer: Convolution-augmented Transformer for Speech Recognition"
authors: "Gulati et al."
year: 2020
venue: "Interspeech 2020"
arxiv: "2005.08100"
url: "https://arxiv.org/abs/2005.08100"
track: speech
tier: extended
est_hours: 4
status: stub
summary: "The dominant ASR encoder block: attention for global and convolution for local context."
related_course: "speech: ASR systems"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
  - id: ctc
    reason: "Blank symbol and collapse mapping"
background: [depthwise-convolution, relative-position]
course_links:
  - title: "Conformers and SpecAugment"
    url: "https://speech.ifkash.dev/lessons/035-conformers-and-specaugment"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Macaron-style feed-forward sandwich
- Convolution module after self-attention
- Relative positional attention
