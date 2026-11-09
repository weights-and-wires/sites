---
id: "soundstream"
title: "SoundStream: An End-to-End Neural Audio Codec"
authors: "Zeghidour et al."
year: 2021
venue: "IEEE/ACM TASLP 2022"
arxiv: "2107.03312"
url: "https://arxiv.org/abs/2107.03312"
track: speech
tier: extended
est_hours: 4
status: stub
summary: "Introduces residual vector quantization for audio tokens."
related_course: "speech: neural codecs"
prereqs:
  - id: gan
    reason: "Generator and discriminator minimax game"
  - id: wavenet
    reason: "Dilated causal convolutions"
background: [residual-vector-quantization, adversarial-losses]
course_links:
  - title: "Neural audio codecs"
    url: "https://speech.ifkash.dev/lessons/050-neural-audio-codecs"
  - title: "Residual and scalar quantization"
    url: "https://speech.ifkash.dev/lessons/049-residual-and-scalar-quantization"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Encoder, RVQ, decoder
- Adversarial and reconstruction losses
- Quantizer dropout for variable bitrate
