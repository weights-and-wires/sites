---
id: transformer
title: "Attention Is All You Need"
authors: "Vaswani et al."
year: 2017
venue: "NeurIPS 2017"
arxiv: "1706.03762"
track: architectures
tier: core
hours: 6
status: stub
tldr: "The Transformer replaces recurrence with self-attention, so every token can read every other token in one step. That makes training parallel across the sequence and became the architecture behind modern language models."
---

## Page in progress

This is a placeholder page. It shows the layout that every paper page uses.
The full page adds the prerequisite chain, worked examples, and exercises that
run in your browser.

## The key idea

Each token builds a query, a key, and a value. Attention weights every value by
how well its key matches the query, scaled by the key width $d_k$:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The following NumPy function computes the same thing for one head:

```python
import numpy as np

def attention(q, k, v):
    scores = q @ k.T / np.sqrt(k.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)  # numerically stable softmax
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v
```

## Further reading

- [The paper on arXiv](https://arxiv.org/abs/1706.03762)
