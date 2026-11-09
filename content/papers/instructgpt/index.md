---
id: "instructgpt"
title: "Training language models to follow instructions with human feedback"
authors: "Ouyang et al."
year: 2022
venue: "NeurIPS 2022"
arxiv: "2203.02155"
url: "https://arxiv.org/abs/2203.02155"
track: rl
tier: core
est_hours: 6
status: stub
summary: "The three-stage SFT, reward model, PPO pipeline behind ChatGPT."
related_course: "rl: RLHF post-training"
prereqs:
  - id: christiano
    reason: "Reward model from pairwise comparisons"
  - id: gpt3
    reason: "Scale to 175B parameters"
  - id: ppo
    reason: "Clipped surrogate objective"
background: [kl-penalty, sft, reward-model]
course_links:
  - title: "RLHF with PPO"
    url: "https://rl.ifkash.dev/lessons/053-rlhf-with-ppo"
  - title: "Supervised fine-tuning and the post-training pipeline"
    url: "https://rl.ifkash.dev/lessons/049-sft-and-the-post-training-pipeline"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- SFT then reward model then PPO
- KL penalty against the SFT policy
- 1.3B aligned model preferred over 175B GPT-3
