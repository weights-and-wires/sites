---
id: "f5tts"
title: "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"
authors: "Chen et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2410.06885"
url: "https://arxiv.org/abs/2410.06885"
track: speech
tier: extended
est_hours: 4
status: stub
summary: "Non-autoregressive TTS with flow matching and a DiT backbone, no duration model."
related_course: "speech: flow-matching TTS"
prereqs:
  - id: flow-matching
    reason: "Regress a conditional vector field"
  - id: dit
    reason: "Latent patches as tokens"
background: [infilling, mel-spectrograms]
course_links:
  - title: "Flow matching for speech"
    url: "https://speech.ifkash.dev/lessons/058-flow-matching-for-speech"
  - title: "Flow matching"
    url: "https://speech.ifkash.dev/lessons/056-flow-matching"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Text padded with filler tokens to the audio length
- Infilling objective
- Sway sampling for inference
