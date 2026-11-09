---
id: "valle"
title: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
authors: "Wang et al."
year: 2023
venue: "arXiv preprint"
arxiv: "2301.02111"
url: "https://arxiv.org/abs/2301.02111"
track: speech
tier: core
est_hours: 5
status: stub
summary: "Treats TTS as codec language modeling with a 3-second voice prompt."
related_course: "speech: TTS"
prereqs:
  - id: encodec
    reason: "Streaming encoder-decoder with RVQ"
  - id: audiolm
    reason: "Semantic tokens from w2v-BERT"
background: [in-context-prompting, autoregressive-and-non-autoregressive-decoding]
course_links:
  - title: "Codec language models"
    url: "https://speech.ifkash.dev/lessons/051-codec-language-models"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- AR model for first codebook, NAR for the rest
- Acoustic prompt gives speaker identity
- Scale to 60k hours
