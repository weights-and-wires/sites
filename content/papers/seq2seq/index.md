---
id: "seq2seq"
title: "Sequence to Sequence Learning with Neural Networks"
authors: "Sutskever et al."
year: 2014
venue: "NeurIPS 2014"
arxiv: "1409.3215"
url: "https://arxiv.org/abs/1409.3215"
track: nlp-llm
tier: extended
est_hours: 3
status: stub
summary: "Defines the encoder-decoder framing that attention and Transformers then fix."
related_course: "ml: sequence models"
prereqs:
  - id: lstm
    reason: "Additive cell state avoids vanishing gradients"
background: [cross-entropy, teacher-forcing, beam-search]
course_links:
  - title: "Recurrent cells and backpropagation through time"
    url: "https://ml.ifkash.dev/lessons/041-recurrent-cells-and-bptt"
  - title: "Attention encoder-decoders"
    url: "https://speech.ifkash.dev/lessons/032-attention-encoder-decoders"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Encode the source into a fixed vector, decode autoregressively
- Reversing the source sentence helped optimization
- Fixed-size bottleneck limits long inputs
