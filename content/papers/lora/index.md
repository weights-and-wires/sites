---
id: "lora"
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: "Hu et al."
year: 2021
venue: "ICLR 2022"
arxiv: "2106.09685"
url: "https://arxiv.org/abs/2106.09685"
track: training-optimization
tier: core
est_hours: 3
status: stub
summary: "The standard way to fine-tune big models on small hardware; also the key to cheap RL post-training."
related_course: "ml: parameter-efficient fine-tuning"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [low-rank-factorization, fine-tuning, matrix-rank]
course_links:
  - title: "Fine-tuning on a budget with LoRA"
    url: "https://ml.ifkash.dev/lessons/057-fine-tuning-and-lora"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Freeze W and learn a low-rank update BA
- Initialize B at zero so training starts at the base model
- Merge the update into W at inference for zero overhead
