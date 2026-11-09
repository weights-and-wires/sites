---
id: "bert"
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors: "Devlin et al."
year: 2018
venue: "NAACL 2019"
arxiv: "1810.04805"
url: "https://arxiv.org/abs/1810.04805"
track: nlp-llm
tier: extended
est_hours: 3
status: stub
summary: "The encoder-only counterpart to GPT; explains why decoder-only won for generation."
related_course: "ml: transformers"
prereqs:
  - id: transformer
    reason: "Scaled dot-product self-attention"
background: [masked-language-modeling, fine-tuning]
course_links:
  - title: "The transformer block: SwiGLU, residuals, and layer norm"
    url: "https://ml.ifkash.dev/lessons/050-the-transformer-block"
  - title: "Scaled attention, many heads, and the causal mask"
    url: "https://ml.ifkash.dev/lessons/047-scaled-attention-and-heads"
---

## Page in progress

This page is a placeholder. The full page adds worked examples and exercises that run in your browser. Until then, the prerequisite panel shows what to read first.

## What the full page covers

- Masked language modeling
- Next-sentence prediction (later dropped)
- Pretrain then fine-tune
