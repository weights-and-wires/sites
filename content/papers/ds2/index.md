---
id: "ds2"
title: "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin"
authors: "Amodei et al."
year: 2015
venue: "ICML 2016"
arxiv: "1512.02595"
url: "https://arxiv.org/abs/1512.02595"
track: speech
tier: extended
est_hours: 4
status: stub
summary: "A scaled end-to-end ASR system and a landmark of engineering for CTC at scale."
related_course: "speech: ASR systems"
prereqs:
  - id: ctc
    reason: "Blank symbol and collapse mapping"
background: [spectrograms, batch-norm-for-rnns]
course_links:
  - title: "An ASR training recipe"
    url: "https://speech.ifkash.dev/lessons/040-asr-recipe"
  - title: "Neural acoustic models"
    url: "https://speech.ifkash.dev/lessons/027-neural-acoustic-models"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- CNN plus bidirectional RNN with CTC
- Data and compute scaling
- Language model rescoring
