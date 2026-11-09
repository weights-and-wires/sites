---
id: "whisper"
title: "Robust Speech Recognition via Large-Scale Weak Supervision"
authors: "Radford et al."
year: 2022
venue: "ICML 2023"
arxiv: "2212.04356"
url: "https://arxiv.org/abs/2212.04356"
track: speech
tier: core
est_hours: 5
status: stub
summary: "A 680k-hour weakly supervised encoder-decoder; the practical default for ASR."
related_course: "speech: ASR systems"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
  - id: bpe
    reason: "Iteratively merge the most frequent symbol pair"
  - id: wav2vec2
    reason: "CNN feature encoder plus Transformer context network"
background: [log-mel-features, multitask-token-format]
course_links:
  - title: "Whisper and large-scale recognition"
    url: "https://speech.ifkash.dev/lessons/037-whisper"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Large noisy weak supervision rather than self-supervision
- Multitask format via special tokens
- Encoder-decoder Transformer on 30 s windows
