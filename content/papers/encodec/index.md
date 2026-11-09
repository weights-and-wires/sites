---
id: "encodec"
title: "High Fidelity Neural Audio Compression"
authors: "Defossez et al."
year: 2022
venue: "TMLR 2023"
arxiv: "2210.13438"
url: "https://arxiv.org/abs/2210.13438"
track: speech
tier: core
est_hours: 4
status: stub
summary: "The open codec that supplies the tokens for VALL-E and many speech LLMs."
related_course: "speech: neural codecs"
prereqs:
  - id: soundstream
    reason: "Encoder, RVQ, decoder"
background: [codec-bitrate, multi-scale-spectral-loss]
course_links:
  - title: "Neural audio codecs"
    url: "https://speech.ifkash.dev/lessons/050-neural-audio-codecs"
  - title: "Residual and scalar quantization"
    url: "https://speech.ifkash.dev/lessons/049-residual-and-scalar-quantization"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Streaming encoder-decoder with RVQ
- Multi-scale STFT discriminator
- Loss balancer
