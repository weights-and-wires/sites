---
id: "deepseek-v3"
title: "DeepSeek-V3 Technical Report"
authors: "DeepSeek-AI et al."
year: 2024
venue: "arXiv preprint"
arxiv: "2412.19437"
url: "https://arxiv.org/abs/2412.19437"
track: architectures
tier: core
est_hours: 5
status: stub
summary: "A full recipe for a frontier open MoE: MLA, auxiliary-loss-free balancing, FP8 training, multi-token prediction."
related_course: "ml: frontier training recipe"
prereqs:
  - id: deepseek-v2
    reason: "Latent KV compression with up-projection"
  - id: fp8
    reason: "E4M3 for weights and activations, E5M2 for gradients"
background: [multi-token-prediction, load-balancing]
course_links:
  - title: "Mixtures of experts: routing, the router's gradient, and balance"
    url: "https://ml.ifkash.dev/lessons/060-mixtures-of-experts"
  - title: "Reasoning with verifiable rewards"
    url: "https://rl.ifkash.dev/lessons/057-reasoning-with-verifiable-rewards"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Auxiliary-loss-free load balancing via bias terms
- FP8 mixed-precision training at scale
- Multi-token prediction objective
