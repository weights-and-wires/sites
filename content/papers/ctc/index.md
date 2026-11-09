---
id: "ctc"
title: "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"
authors: "Graves et al."
year: 2006
venue: "ICML 2006"
arxiv: null
url: "https://www.cs.toronto.edu/~graves/icml_2006.pdf"
track: speech
tier: core
est_hours: 6
status: stub
summary: "The loss that made end-to-end ASR possible without frame-level labels."
related_course: "speech: sequence losses"
prereqs:
  - id: lstm
    reason: "Additive cell state avoids vanishing gradients"
background: [dynamic-programming, forward-backward-algorithm, alignment]
course_links:
  - title: "CTC: many paths, one label"
    url: "https://speech.ifkash.dev/lessons/028-ctc-many-paths-one-label"
  - title: "Computing CTC efficiently"
    url: "https://speech.ifkash.dev/lessons/029-ctc-forward-backward"
  - title: "Decoding CTC"
    url: "https://speech.ifkash.dev/lessons/030-decoding-ctc"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Blank symbol and collapse mapping
- Sum over all alignments with dynamic programming
- Conditional independence across frames
