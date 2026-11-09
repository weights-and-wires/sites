---
id: "bahdanau"
title: "Neural Machine Translation by Jointly Learning to Align and Translate"
authors: "Bahdanau et al."
year: 2014
venue: "ICLR 2015"
arxiv: "1409.0473"
url: "https://arxiv.org/abs/1409.0473"
track: architectures
tier: core
est_hours: 3
status: stub
summary: "The origin of attention: removes the fixed-vector bottleneck by letting the decoder look back."
related_course: "ml: attention"
prereqs:
  - id: seq2seq
    reason: "Encode the source into a fixed vector, decode autoregressively"
background: [softmax, alignment, weighted-average]
course_links:
  - title: "Attention as a soft lookup, with a dot product or a learned scorer"
    url: "https://ml.ifkash.dev/lessons/045-attention-as-soft-lookup"
  - title: "Attention gradients and the alignment matrix"
    url: "https://ml.ifkash.dev/lessons/046-attention-gradients-and-alignment"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Additive (MLP) alignment scores
- Context vector is a softmax-weighted sum of encoder states
- Soft alignment is differentiable
