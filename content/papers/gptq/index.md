---
id: "gptq"
title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
authors: "Frantar et al."
year: 2022
venue: "ICLR 2023"
arxiv: "2210.17323"
url: "https://arxiv.org/abs/2210.17323"
track: inference
tier: core
est_hours: 5
status: stub
summary: "The standard one-shot weight quantization algorithm using second-order information."
related_course: "inference: quantization"
prereqs:
  - id: llm-int8
    reason: "Vector-wise absmax quantization"
background: [hessian, least-squares, layer-wise-quantization]
course_links:
  - title: "Quantization on Ampere"
    url: "https://qwen.ifkash.dev/c/18-quantization"
  - title: "Quantized kernels"
    url: "https://gpu.ifkash.dev/c/18-quantized-kernels"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Layer-wise reconstruction objective
- Quantize columns in order and update the remaining weights
- Lazy batch updates and Cholesky for speed
