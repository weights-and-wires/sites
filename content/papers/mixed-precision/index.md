---
id: "mixed-precision"
title: "Mixed Precision Training"
authors: "Micikevicius et al."
year: 2017
venue: "ICLR 2018"
arxiv: "1710.03740"
url: "https://arxiv.org/abs/1710.03740"
track: training-optimization
tier: core
est_hours: 2
status: stub
summary: "Introduces FP16 compute with FP32 master weights and loss scaling, the basis of all low-precision training."
related_course: "gpu: numerics"
prereqs: []
background: [ieee-754-floating-point, underflow-and-overflow]
course_links:
  - title: "Training kernels"
    url: "https://gpu.ifkash.dev/c/19-training-kernels"
  - title: "Floating point on the GPU"
    url: "https://gpu.ifkash.dev/c/09-floating-point-on-the-gpu"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- FP32 master copy of the weights
- Loss scaling keeps small gradients from underflowing
- Accumulate in higher precision
