---
id: "gpt2"
title: "Language Models are Unsupervised Multitask Learners"
authors: "Radford et al."
year: 2019
venue: "OpenAI technical report"
arxiv: null
url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"
track: nlp-llm
tier: core
est_hours: 6
status: stub
summary: "The decoder-only recipe that later models scale up; small enough to reimplement."
related_course: "ml: GPT from scratch"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
  - id: bpe
    reason: "Iteratively merge the most frequent symbol pair"
background: [autoregressive-language-modeling, perplexity]
course_links:
  - title: "Training a language model: next-token loss, schedules, and clipping"
    url: "https://ml.ifkash.dev/lessons/052-next-token-loss-and-safe-steps"
  - title: "Train a tiny transformer to reverse digits"
    url: "https://ml.ifkash.dev/lessons/053p-project-train-a-tiny-transformer"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Decoder-only Transformer with pre-LN
- Byte-level BPE
- Zero-shot task transfer from next-token prediction
