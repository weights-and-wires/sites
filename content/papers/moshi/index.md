---
id: "moshi"
title: "Moshi: a speech-text foundation model for real-time dialogue"
authors: "Defossez et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2410.00037"
url: "https://arxiv.org/abs/2410.00037"
track: speech
tier: core
est_hours: 5
status: stub
summary: "A full-duplex speech LLM with the Mimi codec and text-token inner monologue."
related_course: "speech: speech LLMs"
prereqs:
  - id: audiolm
    reason: "Semantic tokens from w2v-BERT"
  - id: encodec
    reason: "Streaming encoder-decoder with RVQ"
  - id: llama
    reason: "Pre-norm with RMSNorm"
background: [full-duplex, inner-monologue, streaming]
course_links:
  - title: "Full-duplex models and voice agents"
    url: "https://speech.ifkash.dev/lessons/065-full-duplex-and-voice-agents"
  - title: "Spoken language models"
    url: "https://speech.ifkash.dev/lessons/064-spoken-language-models"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Mimi streaming codec at 12.5 Hz
- Inner monologue: predict text before audio
- Multi-stream modeling of both speakers
