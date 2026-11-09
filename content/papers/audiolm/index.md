---
id: "audiolm"
title: "AudioLM: a Language Modeling Approach to Audio Generation"
authors: "Borsos et al."
year: 2022
venue: "IEEE/ACM TASLP 2023"
arxiv: "2209.03143"
url: "https://arxiv.org/abs/2209.03143"
track: speech
tier: extended
est_hours: 4
status: stub
summary: "Casts audio generation as language modeling over semantic and acoustic tokens."
related_course: "speech: speech language models"
prereqs:
  - id: encodec
    reason: "Streaming encoder-decoder with RVQ"
  - id: wav2vec2
    reason: "CNN feature encoder plus Transformer context network"
  - id: gpt2
    reason: "Decoder-only Transformer with pre-LN"
background: [semantic-vs-acoustic-tokens, hierarchical-modeling]
course_links:
  - title: "Codec language models"
    url: "https://speech.ifkash.dev/lessons/051-codec-language-models"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Semantic tokens from w2v-BERT
- Acoustic tokens from a codec
- Three-stage hierarchical modeling
